import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .evaluation import iou, questions


def features(candidate):
    def reciprocal(rank):
        return 1.0 / rank if rank else 0.0

    base = (
        candidate.similarity,
        reciprocal(candidate.dense_rank),
        reciprocal(candidate.word_rank),
        reciprocal(candidate.trigram_rank),
        math.log1p(candidate.scene.end - candidate.scene.start),
    )
    return base if candidate.cross_logit is None else (*base, candidate.cross_logit)


@dataclass(frozen=True)
class LinearRanker:
    weights: tuple[float, ...]
    intercept: float

    @classmethod
    def load(cls, path):
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(payload["weights"]), payload["intercept"])

    def score(self, candidate):
        return self.intercept + sum(
            weight * value for weight, value in zip(self.weights, features(candidate), strict=True)
        )

    def save(self, path):
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"weights": self.weights, "intercept": self.intercept}, stream)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def training_rows(candidates, references):
    positives, same_video, other_video = [], [], []
    relevant_videos = {scene.video for scene in references}
    for candidate in candidates:
        overlap = max(iou(candidate.scene, reference) for reference in references)
        if overlap >= 0.5:
            positives.append(candidate)
        elif overlap == 0.0:
            negatives = same_video if candidate.scene.video in relevant_videos else other_video
            negatives.append(candidate)
    negatives = same_video[:4] + other_video[:4]
    if positives and negatives:
        yield from ((features(candidate), 1) for candidate in positives[:4])
        yield from ((features(candidate), 0) for candidate in negatives)


def train(data, search, output):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = [
        pair
        for question in questions(data, validation=False)
        for pair in training_rows(search.ranking_pool(question.text)[0], question.references)
    ]
    if not rows:
        raise ValueError("No training questions had both a retrieved positive and a hard negative")
    samples, labels = zip(*rows)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(samples)
    model = LogisticRegression(class_weight="balanced", max_iter=300).fit(standardized, labels)
    weights = model.coef_[0] / scaler.scale_
    intercept = model.intercept_[0] - weights @ scaler.mean_
    LinearRanker(tuple(weights.tolist()), float(intercept)).save(output)
    return len(rows)
