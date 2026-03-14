import hashlib
import math
from dataclasses import dataclass
from time import perf_counter_ns

from .data import Scene, name, read_rows

RANKS = (1, 2, 3, 5)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    references: tuple[Scene, ...]


def questions(data, *, validation):
    texts = {}
    references = {}
    for row in read_rows(data / "train" / "train_qa.csv"):
        question_id = row["question_id"]
        partition = hashlib.sha256(question_id.encode("utf-8")).digest()
        if (int.from_bytes(partition[:4], "big") % 5 == 0) != validation:
            continue
        start, end = float(row["start"]), float(row["end"])
        video = name(row["video_file"])
        if not video or not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid reference for question {question_id!r}")
        references.setdefault(question_id, []).append(Scene(video, start, end))
        variants = texts.setdefault(question_id, {})
        for column in ("question_ru", "question_en"):
            text = (row[column] or "").strip()
            if text:
                variants[text] = None
    return [
        Question(question_id, text, tuple(references[question_id]))
        for question_id, variants in texts.items()
        for text in variants
    ]


def iou(first, second):
    if first.video != second.video:
        return 0.0
    intersection = min(first.end, second.end) - max(first.start, second.start)
    if intersection <= 0:
        return 0.0
    return intersection / ((first.end - first.start) + (second.end - second.start) - intersection)


def metrics(examples):
    video_hits = dict.fromkeys(RANKS, 0)
    scene_hits = dict.fromkeys(RANKS, 0)
    count = 0
    for predicted, references in examples:
        count += 1
        videos = {scene.video for scene in references}
        video_rank = next(
            (rank for rank, scene in enumerate(predicted, 1) if scene.video in videos),
            math.inf,
        )
        scene_rank = next(
            (rank for rank, scene in enumerate(predicted, 1)
             if any(iou(scene, ref) >= 0.5 for ref in references)),
            math.inf,
        )
        for rank in RANKS:
            video_hits[rank] += video_rank <= rank
            scene_hits[rank] += scene_rank <= rank
    if count == 0:
        raise ValueError("No labeled questions in the validation split")
    return {
        "queries": count,
        **{f"VideoRecall@{rank}": video_hits[rank] / count for rank in RANKS},
        **{f"SuccessRate@{rank}": scene_hits[rank] / count for rank in RANKS},
        "Composite": sum(video_hits[k] + scene_hits[k] for k in (1, 2, 3)) / (6 * count),
    }


def evaluate(data, search):
    validation = questions(data, validation=True)
    for question in validation[:10]:
        search(question.text)

    latencies = []

    def timed_results():
        for question in validation:
            start = perf_counter_ns()
            predicted = search(question.text)
            latencies.append((perf_counter_ns() - start) / 1_000_000)
            yield predicted, question.references

    results = metrics(timed_results())
    results["p95_latency_ms"] = round(sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1], 2)
    return results
