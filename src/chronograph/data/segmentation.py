"""Build non-overlapping CHB-MIT segment manifests."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mne


@dataclass(frozen=True)
class Seizure:
    file_name: str
    start_seconds: float
    end_seconds: float


def _summary_seizures(summary_path: Path) -> list[Seizure]:
    lines = summary_path.read_text(errors="replace").splitlines()
    seizures: list[Seizure] = []
    current_file = None
    start = None
    for line in lines:
        if match := re.match(r"File Name: (.+)", line):
            current_file = match.group(1).strip()
            start = None
        elif match := re.match(
            r"Seizure(?: \d+)? Start Time: ([0-9.]+) seconds", line
        ):
            start = float(match.group(1))
        elif match := re.match(
            r"Seizure(?: \d+)? End Time: ([0-9.]+) seconds", line
        ):
            if current_file is None or start is None:
                raise ValueError(f"Invalid seizure entry in {summary_path}: {line}")
            seizures.append(Seizure(current_file, start, float(match.group(1))))
    return seizures


@dataclass(frozen=True)
class Segment:
    subject: str
    file_name: str
    start_seconds: float
    end_seconds: float
    label: str
    seizure_id: str


def _file_duration(path: Path) -> float:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    return float(raw.times[-1] + 1 / raw.info["sfreq"])


def _file_start(path: Path) -> datetime | None:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    value = raw.info.get("meas_date")
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recording_offsets(
    files: list[Path], max_gap_seconds: float
) -> dict[str, tuple[float, float]]:
    durations = {path.name: _file_duration(path) for path in files}
    starts = {path.name: _file_start(path) for path in files}
    first_start = next((starts[path.name] for path in files if starts[path.name]), None)
    offsets: dict[str, tuple[float, float]] = {}
    offset = 0.0
    previous_end = None
    for path in files:
        start = starts[path.name]
        if previous_end is not None and start is not None:
            gap = (start - previous_end).total_seconds()
            if gap < 0 or gap > max_gap_seconds:
                offset += max(0.0, gap)
        elif first_start is not None and start is not None:
            offset = max(0.0, (start - first_start).total_seconds())
        duration = durations[path.name]
        offsets[path.name] = (offset, offset + duration)
        if start is not None:
            previous_end = start.timestamp() + duration
            previous_end = datetime.fromtimestamp(previous_end, tz=start.tzinfo)
        else:
            previous_end = None
        offset += duration
    return offsets


def build_subject_segments(
    subject_dir: Path,
    preictal_seconds: float = 1800,
    interictal_buffer_seconds: float = 14400,
    prediction_horizon_seconds: float = 300,
    max_file_gap_seconds: float = 30,
) -> list[Segment]:
    files = sorted(subject_dir.glob("*.edf"))
    seizures = _summary_seizures(
        subject_dir / f"{subject_dir.name}-summary.txt"
    )
    offsets = _recording_offsets(files, max_file_gap_seconds)

    global_seizures = [
        (offsets[s.file_name][0] + s.start_seconds,
         offsets[s.file_name][0] + s.end_seconds,
         s.file_name)
        for s in seizures
        if s.file_name in offsets
    ]
    segments: list[Segment] = []
    for path in files:
        file_start, file_end = offsets[path.name]
        boundaries = {file_start, file_end}
        labels: list[tuple[float, float, str, str]] = []
        for index, (start, end, file_name) in enumerate(global_seizures):
            seizure_id = f"{file_name}:{index + 1}"
            pre_end = start - prediction_horizon_seconds
            pre_start = pre_end - preictal_seconds
            if index:
                pre_start = max(pre_start, global_seizures[index - 1][1])
            labels.extend(
                [(pre_start, pre_end, "preictal", seizure_id),
                 (start, end, "ictal", seizure_id)]
            )
            boundaries.update(
                (
                    pre_start,
                    pre_end,
                    start,
                    end,
                    start - interictal_buffer_seconds,
                    end + interictal_buffer_seconds,
                )
            )
        points = sorted(p for p in boundaries if file_start <= p <= file_end)
        for left, right in zip(points, points[1:]):
            if right <= left:
                continue
            matching = [
                (label, seizure_id)
                for start, end, label, seizure_id in labels
                if left >= start and right <= end
            ]
            if matching:
                label, seizure_id = matching[0]
            else:
                near_seizure = any(
                    left < end + interictal_buffer_seconds
                    and right > start - interictal_buffer_seconds
                    for start, end, _ in global_seizures
                )
                if near_seizure:
                    continue
                label, seizure_id = "interictal", f"interictal:{path.name}:{left:.3f}"
            segments.append(
                Segment(
                    subject_dir.name,
                    path.name,
                    max(0.0, left - file_start),
                    min(file_end - file_start, right - file_start),
                    label,
                    seizure_id,
                )
            )
    return [segment for segment in segments if segment.end_seconds > segment.start_seconds]


def write_manifest(
    subjects: list[Path],
    output: Path,
    prediction_horizon_seconds: float = 300,
    max_file_gap_seconds: float = 30,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "file", "start_seconds", "end_seconds", "label", "seizure_id"]
    counts: dict[tuple[str, str], int] = {
        (subject.name, label): 0
        for subject in subjects
        for label in ("interictal", "preictal", "ictal")
    }
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject in subjects:
            for segment in build_subject_segments(
                subject,
                prediction_horizon_seconds=prediction_horizon_seconds,
                max_file_gap_seconds=max_file_gap_seconds,
            ):
                counts[(segment.subject, segment.label)] = counts.get(
                    (segment.subject, segment.label), 0
                ) + 1
                writer.writerow({
                    "subject": segment.subject,
                    "file": segment.file_name,
                    "start_seconds": f"{segment.start_seconds:.3f}",
                    "end_seconds": f"{segment.end_seconds:.3f}",
                    "label": segment.label,
                    "seizure_id": segment.seizure_id,
                })
    count_output = output.with_name("segment_counts.csv")
    with count_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subject", "label", "segment_count"]
        )
        writer.writeheader()
        for (subject, label), count in sorted(counts.items()):
            writer.writerow(
                {"subject": subject, "label": label, "segment_count": count}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/physionet.org/files/chbmit/1.0.0"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/segments.csv"))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--prediction-horizon-minutes", type=float, default=5)
    parser.add_argument("--max-file-gap-seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.raw_root
    names = args.subjects or ["chb01", "chb02", "chb03", "chb05", "chb06", "chb07", "chb08", "chb10", "chb23", "chb24"]
    write_manifest(
        [root / name for name in names],
        args.output,
        prediction_horizon_seconds=args.prediction_horizon_minutes * 60,
        max_file_gap_seconds=args.max_file_gap_seconds,
    )


if __name__ == "__main__":
    main()
