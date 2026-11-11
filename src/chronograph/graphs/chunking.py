"""Group graph windows into fixed-length samples."""

from __future__ import annotations

import numpy as np


def chunk_starts(
    count: int, length: int, stride: int
) -> range:
    return range(0, max(0, count - length + 1), stride)


def chunks(
    adjacency: np.ndarray,
    node_features: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    length: int,
    stride: int,
    edge_sign: np.ndarray | None = None,
) -> list[dict[str, np.ndarray | str]]:
    output = []
    for start in chunk_starts(len(labels), length, stride):
        end = start + length
        if len(set(segment_ids[start:end].tolist())) != 1:
            continue
        if len(set(labels[start:end].tolist())) != 1:
            continue
        record = {
            "adjacency": adjacency[start:end],
            "node_features": node_features[start:end],
            "label": labels[start],
            "segment_id": segment_ids[start],
        }
        if edge_sign is not None:
            record["edge_sign"] = edge_sign[start:end]
        output.append(record)
    return output
