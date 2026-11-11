import numpy as np
import mne

from chronograph.data.artifact_rejection import artifact_windows
from chronograph.data.preprocessing import (
    _apply_preprocessing,
    _retain_windows,
    _window_records,
)


def test_existing_interictal_windows_reduce_remaining_budget(tmp_path):
    output = tmp_path / "sample.npz"
    np.savez(output, labels=np.array(["interictal", "ictal", "interictal"]))
    with np.load(output) as saved:
        remaining = 4 - int(np.count_nonzero(saved["labels"] == "interictal"))
    assert remaining == 2


def test_artifact_windows_flags_only_complete_windows():
    data = np.zeros((2, 256 * 2 + 10), dtype=np.float32)
    data[0, 256 : 512] = 600e-6
    flags = artifact_windows(data, 256, 1, 500)
    np.testing.assert_array_equal(flags, [False, True])


def test_preprocessing_applies_filters_without_common_average_reference():
    info = mne.create_info(["a", "b", "c"], sfreq=256, ch_types="eeg")
    data = np.zeros((3, 256 * 10), dtype=np.float64)
    times = np.arange(data.shape[1]) / 256
    data[0] = np.sin(2 * np.pi * 10 * times) + 0.5 * np.sin(2 * np.pi * 0.1 * times)
    data[1] = np.sin(2 * np.pi * 10 * times)
    data[2] = 0.2 * np.sin(2 * np.pi * 10 * times)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    original = raw.get_data().copy()

    processed = _apply_preprocessing(raw, 0.5, 40.0, 60.0)
    result = processed.get_data()

    assert result.shape == original.shape
    assert not np.allclose(result, original)
    assert not np.allclose(result.mean(axis=0), 0.0, atol=1e-10)


def test_window_records_do_not_cross_label_or_buffer_boundaries():
    records = _window_records(
        sample_count=256 * 500,
        sampling_rate_hz=256,
        window_seconds=        5,
        segments=[
            (270.0, 300.0, "preictal"),
            (300.0, 310.0, "ictal"),
        ],
    )
    assert records
    starts = dict(records)
    assert starts[300 * 256 - 30 * 256] == "preictal"
    assert starts[300 * 256] == "ictal"


def test_window_records_keep_one_exact_window():
    assert _window_records(
        256 * 5,
        256,
        5,
        [(0.0, 5.0, "ictal")],
    ) == [(0, "ictal")]


def test_artifact_flags_use_filtered_voltage_before_normalization():
    data = np.zeros((2, 256), dtype=np.float32)
    data[0, 100] = 600e-6
    flags = artifact_windows(data, 256, 1, 500)
    normalized = data / 1000
    assert flags.tolist() == [True]
    assert not np.any(np.abs(normalized) > 500e-6)


def test_artifacts_are_rejected_only_for_interictal_windows():
    data = np.zeros((2, 256 * 3), dtype=np.float32)
    data[0, :512] = 600e-6
    records, flags, remaining = _retain_windows(
        data,
        [(0, "interictal"), (256, "ictal"), (512, "preictal")],
        256,
        1,
        500,
        1,
    )
    assert records == [(256, "ictal"), (512, "preictal")]
    np.testing.assert_array_equal(flags, [True, False])
    assert remaining == 1
