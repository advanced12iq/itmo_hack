import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from .data import Scene, windows

EMBEDDER = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64


class DenseEncoder:
    def __init__(self):
        import faiss
        import torch

        if not torch.cuda.is_available() or getattr(faiss, "get_num_gpus", lambda: 0)() < 1:
            raise RuntimeError("CUDA is required for both PyTorch and FAISS. See environment.yml.")
        import faiss.contrib.torch_utils
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.resources = faiss.StandardGpuResources()
        self.model = SentenceTransformer(EMBEDDER, device="cuda:0")
        self.model.max_seq_length = 256

    def encode(self, texts, prefix):
        return self.model.encode(
            [f"{prefix}: {text}" for text in texts],
            batch_size=BATCH_SIZE,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float().contiguous()

    def new_index(self):
        dimension = self.model.get_sentence_embedding_dimension()
        return self.to_gpu(self.faiss.IndexFlatIP(dimension))

    def to_gpu(self, index):
        return self.faiss.index_cpu_to_gpu(self.resources, 0, index)

    def save(self, index, path):
        self.faiss.write_index(self.faiss.index_gpu_to_cpu(index), str(path))

    def load(self, path):
        return self.to_gpu(self.faiss.read_index(str(path)))


def build(data, output):
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new index directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = DenseEncoder()
    with TemporaryDirectory(dir=output.parent, prefix=".okko-") as staging:
        snapshot = Path(staging) / "snapshot"
        snapshot.mkdir()
        with closing(sqlite3.connect(snapshot / "scenes.sqlite")) as db:
            db.execute("CREATE TABLE scenes (id INTEGER PRIMARY KEY, video TEXT, "
                       "start REAL, end REAL, text TEXT, UNIQUE(video, start, end))")
            db.executemany(
                "INSERT OR IGNORE INTO scenes(video, start, end, text) VALUES (?, ?, ?, ?)",
                ((scene.video, scene.start, scene.end, scene.text) for scene in windows(data)),
            )
            count = db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
            if not count:
                raise ValueError("No transcript windows matched video_files.csv")
            index = encoder.new_index()
            rows = db.execute("SELECT text FROM scenes ORDER BY id")
            while batch := rows.fetchmany(BATCH_SIZE):
                index.add(encoder.encode([text for (text,) in batch], "passage"))
            db.commit()
        encoder.save(index, snapshot / "dense.faiss")
        snapshot.rename(output)
    return count


class Search:
    def __init__(self, directory):
        index_file = directory / "dense.faiss"
        database = directory / "scenes.sqlite"
        if not index_file.is_file() or not database.is_file():
            raise FileNotFoundError(f"{directory}: build the search index first")
        self.encoder = DenseEncoder()
        self.index = self.encoder.load(index_file)
        self.db = sqlite3.connect(database)

    def close(self):
        self.db.close()

    def top(self, question):
        if not question.strip():
            return []
        vector = self.encoder.encode([question], "query")
        _, positions = self.index.search(vector, min(5, self.index.ntotal))
        return [
            Scene(*self.db.execute(
                "SELECT video, start, end, text FROM scenes WHERE id = ?", (position + 1,)
            ).fetchone())
            for position in positions[0].tolist() if position >= 0
        ]
