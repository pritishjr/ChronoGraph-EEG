from datetime import datetime, timezone

from chronograph.data.segmentation import Segment
from chronograph.data.segmentation import _recording_offsets, _summary_seizures


def test_segment_labels_are_not_mixed():
    segments = [
        Segment("chb01", "x.edf", 0, 10, "interictal", ""),
        Segment("chb01", "x.edf", 10, 20, "preictal", "x:1"),
        Segment("chb01", "x.edf", 20, 30, "ictal", "x:1"),
    ]
    for left, right in zip(segments, segments[1:]):
        assert left.end_seconds == right.start_seconds
        assert left.label != right.label


def test_summary_parser_accepts_numbered_seizures(tmp_path):
    summary = tmp_path / "chb06-summary.txt"
    summary.write_text(
        "File Name: chb06_01.edf\n"
        "Seizure 1 Start Time: 10 seconds\n"
        "Seizure 1 End Time: 20 seconds\n"
    )
    seizures = _summary_seizures(summary)
    assert [(item.start_seconds, item.end_seconds) for item in seizures] == [(10, 20)]


def test_recording_offsets_preserve_large_file_gaps(monkeypatch, tmp_path):
    files = [tmp_path / "a.edf", tmp_path / "b.edf"]
    starts = {
        files[0].name: datetime(2026, 1, 1, tzinfo=timezone.utc),
        files[1].name: datetime(2026, 1, 1, 0, 0, 40, tzinfo=timezone.utc),
    }
    monkeypatch.setattr(
        "chronograph.data.segmentation._file_duration",
        lambda path: 10.0,
    )
    monkeypatch.setattr(
        "chronograph.data.segmentation._file_start",
        lambda path: starts[path.name],
    )
    offsets = _recording_offsets(files, max_gap_seconds=20)
    assert offsets["a.edf"] == (0.0, 10.0)
    assert offsets["b.edf"] == (40.0, 50.0)
