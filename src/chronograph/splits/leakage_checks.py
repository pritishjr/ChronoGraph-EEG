"""Checks preventing subject and seizure leakage."""

from __future__ import annotations


def assert_no_leakage(
    records: list[dict[str, str]],
    train_ids: list[str],
    validation_ids: list[str],
    test_ids: list[str],
) -> None:
    by_id = {record["id"]: record for record in records}
    partitions = [
        {by_id[item]["id"] for item in ids} for ids in
        (train_ids, validation_ids, test_ids)
    ]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    for left_ids, right_ids in (
        (train_ids, validation_ids),
        (train_ids, test_ids),
        (validation_ids, test_ids),
    ):
        left_groups = {by_id[item]["seizure_id"] for item in left_ids}
        right_groups = {by_id[item]["seizure_id"] for item in right_ids}
        assert not left_groups & right_groups
