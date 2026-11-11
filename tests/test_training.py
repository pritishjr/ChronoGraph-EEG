import numpy as np

from chronograph.training.__main__ import _merge_config
from chronograph.training.trainer import macro_f1, undersample_interictal


def test_training_macro_f1_is_defined_for_empty_classes():
    assert macro_f1(np.array([0, 0]), np.array([0, 1])) > 0


def test_interictal_undersampling_only_changes_training_data():
    values = np.arange(6)[:, None]
    labels = np.array(["interictal"] * 4 + ["ictal"] * 2)
    _, _, sampled = undersample_interictal((values, values, labels), seed=42)
    assert np.count_nonzero(sampled == "interictal") == 2
    assert np.count_nonzero(sampled == "ictal") == 2


def test_model_config_merges_nested_values_without_dropping_base_values():
    merged = _merge_config(
        {"training": {"seed": 42, "epochs": 10}, "model": {"hidden_dim": 16}},
        {"training": {"epochs": 3}, "model": {"eigenvectors": 8}},
    )
    assert merged == {
        "training": {"seed": 42, "epochs": 3},
        "model": {"hidden_dim": 16, "eigenvectors": 8},
    }
