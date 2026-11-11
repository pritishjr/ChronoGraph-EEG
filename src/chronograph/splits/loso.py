"""Build leave-one-subject-out folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .validation_split import split_by_group


def load_chunk_records(processed_root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(processed_root.glob("*/*.npz")):
        with np.load(path) as saved:
            subject = str(saved["subject"])
            source_file = str(saved["source_file"])
            labels = saved["labels"].astype(str)
            segment_ids = saved["segment_ids"].astype(str)
        for index, (label, segment_id) in enumerate(zip(labels, segment_ids)):
            records.append(
                {
                    "id": f"{subject}/{source_file}:{index}",
                    "subject": subject,
                    "seizure_id": f"{subject}:{segment_id}",
                    "label": label,
                }
            )
    return records


def build_folds(
    records: list[dict[str, str]],
    validation_fraction: float = 0.2,
    seed: int = 42,
    subject_groups: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    subject_groups = subject_groups or {"chb01": "patient_01", "chb21": "patient_01"}
    groups = {
        subject: subject_groups.get(subject, subject)
        for subject in {record["subject"] for record in records}
    }
    held_out_groups = sorted(set(groups.values()))
    folds = []
    for held_out in held_out_groups:
        test = [record for record in records if groups[record["subject"]] == held_out]
        training = [
            record for record in records if groups[record["subject"]] != held_out
        ]
        train, validation = split_by_group(
            training, validation_fraction=validation_fraction, seed=seed
        )
        folds.append(
            {
                "fold": held_out,
                "test_subject": held_out,
                "train": [record["id"] for record in train],
                "validation": [record["id"] for record in validation],
                "test": [record["id"] for record in test],
            }
        )
    return folds


def write_folds(
    processed_root: Path,
    output: Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    records = load_chunk_records(processed_root)
    if not records:
        raise RuntimeError(f"No processed graph chunks found in {processed_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "records": records,
                "folds": build_folds(records, validation_fraction, seed),
            },
            indent=2,
        )
        + "\n"
    )
