"""Build Pearson graph chunks from compact P2 windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .chunking import chunks
from .connectivity import pearson
from .node_features import band_power
from .sparsify import top_k


def _segments(path: Path, subject: str, source_file: str) -> list[tuple[float, float, str, str]]:
    with path.open(newline="") as handle:
        return [
            (
                float(row["start_seconds"]),
                float(row["end_seconds"]),
                row["label"],
                row["seizure_id"],
            )
            for row in csv.DictReader(handle)
            if row["subject"] == subject and row["file"] == source_file
        ]


def _segment_ids(
    starts: np.ndarray,
    sampling_rate_hz: float,
    segments: list[tuple[float, float, str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    ids = []
    labels = []
    for start in starts:
        seconds = float(start / sampling_rate_hz)
        matches = [
            (index, label, seizure_id)
            for index, (left, right, label, seizure_id) in enumerate(segments)
            if seconds >= left and seconds + 5 <= right
        ]
        if len(matches) != 1:
            ids.append("")
            labels.append("")
        else:
            index, label, seizure_id = matches[0]
            labels.append(label)
            ids.append(seizure_id or f"interictal:{index}")
    return np.array(labels), np.array(ids)


def build_file(
    source: Path,
    output: Path,
    subject: str,
    manifest: Path,
    sampling_rate_hz: float,
    bands: dict[str, tuple[float, float]],
    top_k_nodes: int,
    chunk_length: int,
    stride: int,
) -> int:
    with np.load(source) as saved:
        windows = saved["data"].astype(np.float32)
        labels = saved["labels"].astype(str)
        starts = saved.get("window_starts")
    if starts is None:
        raise ValueError(
            f"{source} has no window_starts; regenerate it with the current P2 format"
        )
    source_file = source.stem + ".edf"
    manifest_segments = _segments(manifest, subject, source_file)
    manifest_labels, segment_ids = _segment_ids(starts, sampling_rate_hz, manifest_segments)
    valid = manifest_labels != ""
    windows, labels, segment_ids = windows[valid], manifest_labels[valid], segment_ids[valid]
    if not len(windows):
        return 0
    signed = np.stack([pearson(window) for window in windows])
    adjacency = np.stack([top_k(matrix, top_k_nodes) for matrix in signed])
    edge_sign = np.sign(signed).astype(np.int8)
    edge_sign[adjacency == 0] = 0
    features = np.stack([band_power(window, sampling_rate_hz, bands) for window in windows])
    records = chunks(
        adjacency, features, labels, segment_ids, chunk_length, stride, edge_sign
    )
    if not records:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        adjacency=np.stack([record["adjacency"] for record in records]),
        edge_sign=np.stack([record["edge_sign"] for record in records]),
        node_features=np.stack([record["node_features"] for record in records]),
        labels=np.array([record["label"] for record in records]),
        segment_ids=np.array([record["segment_id"] for record in records]),
        subject=subject,
        source_file=source_file,
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/segments.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-length", type=int, default=12)
    parser.add_argument("--chunk-stride", type=int, default=60)
    args = parser.parse_args()
    bands = {
        "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
    }
    for subject_dir in sorted(args.input_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for source in sorted(subject_dir.glob("*.npz")):
            output = args.output_root / subject_dir.name / source.name
            count = build_file(
                source, output, subject_dir.name, args.manifest, 256, bands,
                args.top_k, args.chunk_length, args.chunk_stride,
            )
            print(f"{subject_dir.name}/{source.name}: {count} chunks")


if __name__ == "__main__":
    main()
