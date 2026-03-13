import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Scene:
    video: str
    start: float
    end: float
    text: str = ""


def name(value):
    return PurePosixPath(str(value).strip().replace("\\", "/")).stem


def video_id(value):
    return name(value).removeprefix("video_").removeprefix("audio_")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_catalog(path):
    catalog = {}
    for row in read_rows(path):
        field = next(
            (field for field in ("video_file", "video_path", "video_name", "path",
                                 "file_path", "filename", "file_name") if field in row),
            None,
        )
        if field is None and len(row) == 1:
            field = next(iter(row))
        if field is None:
            raise ValueError(f"{path}: video filename column not found")
        video = name(row[field] or "")
        if video:
            catalog[video_id(video)] = video
    if not catalog:
        raise ValueError(f"{path}: empty video catalog")
    return catalog


def transcript_records(payload):
    if isinstance(payload, dict):
        if "segments" in payload:
            yield payload["audio_file"], payload["segments"]
        else:
            for path, segments in payload.items():
                if isinstance(segments, dict):
                    yield segments.get("audio_file", path), segments["segments"]
                else:
                    yield path, segments
    else:
        for record in payload:
            yield record["audio_file"], record["segments"]


def transcript_segments(data):
    catalog = read_catalog(data / "video_files.csv")
    with (data / "transcripts.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    for path, records in transcript_records(payload):
        video = catalog.get(video_id(path))
        if video is None:
            continue
        segments = []
        for record in records:
            start, end = float(record["start"]), float(record["end"])
            text = " ".join((record.get("text") or "").split())
            if math.isfinite(start) and math.isfinite(end) and 0 <= start < end and text:
                segments.append(Scene(video, start, end, text))
        if segments:
            yield video, sorted(segments, key=lambda segment: (segment.start, segment.end))


def windows(data):
    for video, segments in transcript_segments(data):
        next_anchor = 0.0
        for index, first in enumerate(segments):
            if first.start < next_anchor:
                continue
            next_anchor = first.start + 6
            for duration in (12, 24, 48):
                end = first.end
                parts = [first.text]
                cursor = index + 1
                while cursor < len(segments):
                    segment = segments[cursor]
                    if segment.start >= first.start + duration or segment.start > end + 8:
                        break
                    end = max(end, segment.end)
                    parts.append(segment.text)
                    cursor += 1
                yield Scene(video, first.start, end, " ".join(parts))


def write_submission(data, output, search):
    columns = ["query_id"] + [
        f"{field}_{rank}"
        for rank in range(1, 6)
        for field in ("video_file", "start", "end")
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in read_rows(data / "test" / "test.csv"):
            scenes = search(row["question"] or "")[:5]
            values = [value for scene in scenes for value in (scene.video, scene.start, scene.end)]
            values.extend(["", 0.0, 0.0] * (5 - len(scenes)))
            writer.writerow([row["query_id"], *values])
