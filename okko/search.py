import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .cross import CROSS_LIMIT, CrossEncoder
from .data import Scene, windows
from .lexical import build_lexical, lexical_hits
from .rerank import LinearRanker

EMBEDDER = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64


@dataclass
class Candidate:
    id: int
    scene: Scene
    similarity: float = 0.0
    dense_rank: int = 0
    word_rank: int = 0
    trigram_rank: int = 0
    cross_logit: float | None = None

    @property
    def fused_score(self):
        return sum(
            weight / (60 + rank)
            for rank, weight in (
                (self.dense_rank, 1.0), (self.word_rank, 0.65), (self.trigram_rank, 0.45)
            )
            if rank
        )


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
            build_lexical(db)
            db.commit()
        encoder.save(index, snapshot / "dense.faiss")
        snapshot.rename(output)
    return count


class Search:
    def __init__(self, directory, *, cross=True):
        index_file = directory / "dense.faiss"
        database = directory / "scenes.sqlite"
        if not index_file.is_file() or not database.is_file():
            raise FileNotFoundError(f"{directory}: build the search index first")
        self.encoder = DenseEncoder()
        self.index = self.encoder.load(index_file)
        self.cross = CrossEncoder() if cross else None
        self.ranker_path = directory / ("ranker-cross.json" if cross else "ranker.json")
        self.ranker = LinearRanker.load(self.ranker_path)
        self.db = sqlite3.connect(database)

    def close(self):
        self.db.close()

    def candidates(self, question):
        question = question.strip()
        if not question:
            return []
        vector = self.encoder.encode([question], "query")
        scores, positions = self.index.search(vector, min(50, self.index.ntotal))
        dense = {
            position + 1: (score, rank)
            for rank, (score, position) in enumerate(zip(scores[0].tolist(), positions[0].tolist()), 1)
            if position >= 0
        }
        word_hits, trigram_hits = lexical_hits(self.db, question)
        word_ranks = {rowid: rank for rank, rowid in enumerate(word_hits, 1)}
        trigram_ranks = {rowid: rank for rank, rowid in enumerate(trigram_hits, 1)}
        ids = list(dense.keys() | word_ranks.keys() | trigram_ranks.keys())
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.execute(
            f"SELECT id, video, start, end, text FROM scenes WHERE id IN ({placeholders})", ids
        )
        candidates = [
            Candidate(rowid, Scene(video, start, end, text), *dense.get(rowid, (0.0, 0)),
                      word_ranks.get(rowid, 0), trigram_ranks.get(rowid, 0))
            for rowid, video, start, end, text in rows
        ]
        return sorted(candidates, key=lambda candidate: (-candidate.fused_score, candidate.id))

    def ranking_pool(self, question):
        candidates = self.candidates(question)
        if not self.cross or not candidates:
            return candidates, []
        shortlist, reserve = candidates[:CROSS_LIMIT], candidates[CROSS_LIMIT:]
        for candidate, score in zip(shortlist, self.cross.score(question, shortlist), strict=True):
            candidate.cross_logit = score
        return shortlist, reserve

    def top(self, question):
        shortlist, reserve = self.ranking_pool(question)
        if self.ranker:
            shortlist.sort(key=self.ranker.score, reverse=True)
        elif self.cross:
            cross_order = sorted(shortlist, key=lambda candidate: candidate.cross_logit, reverse=True)
            rerank_scores = {
                candidate.id: 1.25 / (60 + rank)
                for rank, candidate in enumerate(cross_order, 1)
            }
            shortlist.sort(
                key=lambda candidate: candidate.fused_score + rerank_scores[candidate.id],
                reverse=True,
            )
        return [candidate.scene for candidate in (shortlist + reserve)[:5]]
