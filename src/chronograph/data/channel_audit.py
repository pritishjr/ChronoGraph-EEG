"""Audit CHB-MIT EDF headers and select a stable channel set."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdfHeader:
    channels: tuple[str, ...]
    sampling_rate_hz: float


def _field(data: bytes, start: int, width: int) -> str:
    return data[start : start + width].decode("ascii", "replace").strip()


def read_edf_header(path: Path) -> EdfHeader:
    """Read the fixed EDF header and signal labels without loading samples."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF header is shorter than 256 bytes")
        header_bytes = int(_field(fixed, 184, 8))
        signal_count = int(_field(fixed, 252, 4))
        header = handle.read(header_bytes - 256)

    if len(header) != header_bytes - 256:
        raise ValueError("EDF signal header is incomplete")

    labels = tuple(_field(header, index * 16, 16) for index in range(signal_count))
    samples_start = signal_count * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_per_record = tuple(
        int(_field(header, samples_start + index * 8, 8))
        for index in range(signal_count)
    )
    duration_start = 236
    record_duration = float(_field(fixed, duration_start, 8))
    if record_duration <= 0 or len(set(samples_per_record)) != 1:
        raise ValueError("EDF has invalid or mixed signal timing")

    return EdfHeader(
        channels=labels,
        sampling_rate_hz=samples_per_record[0] / record_duration,
    )


def _subject_dirs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("chb*") if path.is_dir())


def _seizure_records(raw_dir: Path) -> set[str]:
    records_file = raw_dir / "RECORDS-WITH-SEIZURES"
    if not records_file.exists():
        return set()
    return {
        line.strip()
        for line in records_file.read_text().splitlines()
        if line.strip()
    }


def audit_raw_dataset(raw_dir: Path, output_csv: Path) -> dict[str, object]:
    """Audit all EDF files and write one row per subject, file, and channel."""
    subject_dirs = _subject_dirs(raw_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No CHB-MIT subject directories found in {raw_dir}")

    records_with_seizures = _seizure_records(raw_dir)
    headers: dict[Path, EdfHeader] = {}
    subject_sets: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    errors: list[tuple[str, str]] = []

    for subject_dir in subject_dirs:
        for edf_path in sorted(subject_dir.glob("*.edf")):
            try:
                header = read_edf_header(edf_path)
            except (OSError, ValueError) as exc:
                errors.append((str(edf_path), str(exc)))
                continue
            headers[edf_path] = header
            subject_sets[subject_dir.name][header.channels] += 1

    stable_sets = {
        subject: next(iter(channel_sets))
        for subject, channel_sets in subject_sets.items()
        if len(channel_sets) == 1
    }
    groups: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for subject, channels in stable_sets.items():
        groups[channels].append(subject)
    if not groups:
        raise RuntimeError("No subject has one stable channel layout")

    canonical_channels, selected_subjects = max(
        groups.items(), key=lambda item: (len(item[1]), len(item[0]))
    )
    selected_subjects = tuple(sorted(selected_subjects))
    selected_set = set(selected_subjects)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "file",
                "channel",
                "canonical",
                "selected_subject",
                "exclusion_reason",
                "sampling_rate_hz",
                "header_ok",
                "seizure_annotation",
            ],
        )
        writer.writeheader()
        for edf_path, header in sorted(headers.items()):
            subject = edf_path.parent.name
            selected = subject in selected_set
            if selected:
                reason = ""
            elif subject not in stable_sets:
                reason = "channel layout changes across files"
            else:
                reason = "stable channel layout differs from canonical set"
            relative_file = edf_path.relative_to(raw_dir)
            annotation_key = str(relative_file)
            has_annotation = edf_path.with_name(edf_path.name + ".seizures").exists()
            for channel in header.channels:
                writer.writerow(
                    {
                        "subject": subject,
                        "file": str(edf_path.relative_to(raw_dir)),
                        "channel": channel,
                        "canonical": channel in canonical_channels,
                        "selected_subject": selected,
                        "exclusion_reason": reason,
                        "sampling_rate_hz": header.sampling_rate_hz,
                        "header_ok": True,
                        "seizure_annotation": has_annotation,
                    }
                )

    missing_annotations = [
        str(path.relative_to(raw_dir))
        for path in headers
        if str(path.relative_to(raw_dir)) in records_with_seizures
        and not path.with_name(path.name + ".seizures").exists()
    ]
    if errors:
        details = []
        details.append(f"{len(errors)} EDF header errors")
        raise RuntimeError("; ".join(details))

    return {
        "subjects": len(subject_dirs),
        "edf_files": len(headers),
        "selected_subjects": selected_subjects,
        "canonical_channels": canonical_channels,
        "excluded_subjects": tuple(
            subject for subject in sorted(subject_sets) if subject not in selected_set
        ),
        "annotation_warnings": tuple(missing_annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/physionet.org/files/chbmit/1.0.0"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/channel_audit.csv"),
    )
    args = parser.parse_args()
    result = audit_raw_dataset(args.raw_dir, args.output)
    print(f"Audited {result['edf_files']} EDF files across {result['subjects']} subjects.")
    print(f"Selected subjects: {', '.join(result['selected_subjects'])}")
    print(f"Canonical channels: {len(result['canonical_channels'])}")
    for warning in result["annotation_warnings"]:
        print(f"Warning: seizure index lists {warning}, but no sidecar exists.")


if __name__ == "__main__":
    main()
