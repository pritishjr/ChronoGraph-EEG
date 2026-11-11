"""Preprocess selected CHB-MIT EEG files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mne
import numpy as np

def _window_records(
    sample_count: int,
    sampling_rate_hz: float,
    window_seconds: float,
    segments: list[tuple[float, float, str]],
) -> list[tuple[int, str]]:
    """Return windows fully contained in one manifest segment."""
    size = int(round(sampling_rate_hz * window_seconds))
    step = size // 2
    records: list[tuple[int, str]] = []
    for segment_start, segment_end, label in segments:
        first = int(round(segment_start * sampling_rate_hz))
        last = int(round(segment_end * sampling_rate_hz))
        for start in range(first, last - size + 1, step):
            if start + size > last:
                continue
            records.append((start, label))
    return records


def _retain_windows(
    data: np.ndarray,
    records: list[tuple[int, str]],
    sampling_rate_hz: float,
    window_seconds: float,
    amplitude_limit_uv: float,
    interictal_windows_left: int,
) -> tuple[list[tuple[int, str]], np.ndarray, int]:
    sample_count = int(round(sampling_rate_hz * window_seconds))
    flags = np.array(
        [
            np.max(np.abs(data[:, start:start + sample_count]))
            > amplitude_limit_uv * 1e-6
            for start, _ in records
        ],
        dtype=bool,
    )
    retained = []
    retained_flags = []
    for record, flagged in zip(records, flags):
        if record[1] == "interictal":
            if flagged or interictal_windows_left <= 0:
                continue
            interictal_windows_left -= 1
        retained.append(record)
        retained_flags.append(flagged)
    return retained, np.array(retained_flags, dtype=bool), interictal_windows_left


def _load_segments(
    manifest_path: Path, subject: str
) -> dict[str, list[tuple[float, float, str]]]:
    segments: dict[str, list[tuple[float, float, str]]] = {}
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subject"] != subject:
                continue
            segments.setdefault(row["file"], []).append(
                (float(row["start_seconds"]), float(row["end_seconds"]), row["label"])
            )
    return segments


def _subject_stats(
    files: list[Path],
    channel_count: int,
    sampling_rate_hz: float,
    filter_low_hz: float,
    filter_high_hz: float,
    notch_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sums = np.zeros(channel_count, dtype=np.float64)
    squares = np.zeros(channel_count, dtype=np.float64)
    for path in files:
        raw = _preprocess_raw(
            path,
            channel_count,
            sampling_rate_hz,
            filter_low_hz,
            filter_high_hz,
            notch_hz,
        )
        data = raw.get_data().astype(np.float32)
        values = data.astype(np.float64)
        total += values.shape[1]
        sums += values.sum(axis=1)
        squares += np.square(values).sum(axis=1)
    if total == 0:
        raise RuntimeError("No samples found for subject")
    mean = sums / total
    variance = np.maximum(squares / total - np.square(mean), 1e-16)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _preprocess_raw(
    path: Path,
    channel_count: int,
    sampling_rate_hz: float,
    filter_low_hz: float,
    filter_high_hz: float,
    notch_hz: float,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if len(raw.ch_names) < channel_count:
        raise ValueError(f"{path} has fewer than {channel_count} channels")
    if not np.isclose(raw.info["sfreq"], sampling_rate_hz):
        raise ValueError(
            f"{path} has sampling rate {raw.info['sfreq']}, "
            f"expected {sampling_rate_hz}"
        )
    raw.pick(list(range(channel_count)))
    return _apply_preprocessing(
        raw, filter_low_hz, filter_high_hz, notch_hz
    )


def _apply_preprocessing(
    raw: mne.io.BaseRaw,
    filter_low_hz: float,
    filter_high_hz: float,
    notch_hz: float,
) -> mne.io.BaseRaw:
    raw.filter(filter_low_hz, filter_high_hz, method="fir", phase="zero", verbose="ERROR")
    raw.notch_filter(notch_hz, phase="zero", verbose="ERROR")
    return raw


def preprocess_subject(
    subject_dir: Path,
    output_dir: Path,
    canonical_channels: list[str],
    sampling_rate_hz: float,
    filter_low_hz: float,
    filter_high_hz: float,
    notch_hz: float,
    amplitude_limit_uv: float,
    window_seconds: float,
    retained_interictal_hours: float,
    segment_manifest: Path,
    log_path: Path,
) -> int:
    files = sorted(subject_dir.glob("*.edf"))
    if not files:
        return 0
    segments = _load_segments(segment_manifest, subject_dir.name)
    stats_files = files
    means, stds = _subject_stats(
        stats_files,
        len(canonical_channels),
        sampling_rate_hz,
        filter_low_hz,
        filter_high_hz,
        notch_hz,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_exists = log_path.exists()
    log_handle = log_path.open("a", newline="")
    log_writer = csv.DictWriter(
        log_handle,
        fieldnames=[
            "subject",
            "source_file",
            "output_file",
            "status",
            "source_bytes",
            "samples",
            "channels",
            "artifact_windows",
            "interictal_samples_for_stats",
        ],
    )
    if not log_exists:
        log_writer.writeheader()
    processed = 0
    interictal_windows_left = int(
        retained_interictal_hours * 3600 / (window_seconds / 2)
    )
    for existing in output_dir.glob("*.npz"):
        with np.load(existing) as saved:
            interictal_windows_left -= int(
                np.count_nonzero(saved["labels"] == "interictal")
            )
    interictal_windows_left = max(0, interictal_windows_left)
    try:
        for path in files:
            output = output_dir / f"{path.stem}.npz"
            if path.name not in segments:
                log_writer.writerow(
                    {
                        "subject": subject_dir.name,
                        "source_file": path.name,
                        "output_file": "",
                        "status": "no_retained_windows",
                        "source_bytes": path.stat().st_size,
                        "samples": "",
                        "channels": "",
                        "artifact_windows": 0,
                        "interictal_samples_for_stats": "",
                    }
                )
                log_handle.flush()
                processed += 1
                continue
            if output.exists():
                log_writer.writerow(
                    {
                        "subject": subject_dir.name,
                        "source_file": path.name,
                        "output_file": output.name,
                        "status": "skipped_existing",
                        "source_bytes": path.stat().st_size,
                    }
                )
                log_handle.flush()
                processed += 1
                continue
            raw = _preprocess_raw(
                path,
                len(canonical_channels),
                sampling_rate_hz,
                filter_low_hz,
                filter_high_hz,
                notch_hz,
            )
            data = raw.get_data().astype(np.float32)
            records = _window_records(
                data.shape[1],
                sampling_rate_hz,
                window_seconds,
                segments.get(path.name, []),
            )
            records, flags, interictal_windows_left = _retain_windows(
                data,
                records,
                sampling_rate_hz,
                window_seconds,
                amplitude_limit_uv,
                interictal_windows_left,
            )
            if not records:
                log_writer.writerow(
                    {
                        "subject": subject_dir.name,
                        "source_file": path.name,
                        "output_file": "",
                        "status": "no_retained_windows",
                        "source_bytes": path.stat().st_size,
                        "samples": data.shape[1],
                        "channels": data.shape[0],
                        "artifact_windows": 0,
                        "interictal_samples_for_stats": "",
                    }
                )
                log_handle.flush()
                processed += 1
                continue
            starts = np.array([start for start, _ in records])
            labels = np.array([label for _, label in records])
            normalized = (data - means[:, None]) / stds[:, None]
            windows = np.stack(
                [normalized[:, start:start + int(sampling_rate_hz * window_seconds)]
                 for start in starts]
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output,
                data=windows.astype(np.float16),
                labels=labels,
                window_starts=starts,
                artifact_windows=flags,
                sampling_rate_hz=np.float32(sampling_rate_hz),
                channel_positions=np.arange(len(canonical_channels), dtype=np.int16),
                subject=subject_dir.name,
                source_file=path.name,
                mean_v=means,
                std_v=stds,
            )
            log_writer.writerow(
                {
                    "subject": subject_dir.name,
                    "source_file": path.name,
                    "output_file": output.name,
                    "status": "processed",
                    "source_bytes": path.stat().st_size,
                    "samples": data.shape[1],
                    "channels": data.shape[0],
                    "artifact_windows": int(flags.sum()),
                    "interictal_samples_for_stats": "",
                }
            )
            log_handle.flush()
            processed += 1
    finally:
        log_handle.close()
    return processed


def _load_config(path: Path) -> dict:
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/physionet.org/files/chbmit/1.0.0"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=Path("data/manifests/segments.csv"),
    )
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    config = _load_config(args.config)
    data_config = config["data"]
    prep = config["preprocessing"]
    subjects = args.subjects or data_config["selected_subjects"]
    for subject in subjects:
        count = preprocess_subject(
            args.raw_root / subject,
            args.output_root / subject,
            data_config["canonical_channels"],
            data_config["sampling_rate_hz"],
            prep["bandpass_hz"][0],
            prep["bandpass_hz"][1],
            prep["notch_hz"],
            prep["amplitude_rejection_uv"],
            config["graphs"]["window_seconds"],
            config["segmentation"]["retained_interictal_hours_per_subject"],
            args.segment_manifest,
            args.output_root / "preprocessing_log.csv",
        )
        print(f"{subject}: {count} files ready")


if __name__ == "__main__":
    main()
