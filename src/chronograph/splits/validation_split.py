"""Seizure-aware validation splitting."""

from __future__ import annotations

from collections import defaultdict
import random


def split_by_group(
    records: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["seizure_id"]].append(record)
    group_names = sorted(groups)
    if len(group_names) < 2:
        return records, []
    validation_count = max(1, round(len(group_names) * validation_fraction))
    shuffled = group_names[:]
    random.Random(seed).shuffle(shuffled)
    validation_names = set(shuffled[:validation_count])
    validation = [record for record in records if record["seizure_id"] in validation_names]
    train = [record for record in records if record["seizure_id"] not in validation_names]
    return train, validation
