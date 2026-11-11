import numpy as np

from chronograph.evaluation.metrics import classification_metrics


def test_classification_metrics_reports_specificity_auroc_and_confusion_matrix():
    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        ["interictal", "preictal"],
        np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]),
    )
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["per_class"]["interictal"]["specificity"] == 1.0
    assert result["per_class"]["interictal"]["auroc"] == 1.0


def test_aggregate_runs_writes_empty_safe_result_tables(tmp_path):
    from chronograph.evaluation.aggregate import aggregate_runs

    run = tmp_path / "runs" / "svm" / "three_class" / "chb01"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"validation_macro_f1": 0.5, "epoch": 1}')
    per_subject, main = aggregate_runs(tmp_path / "runs", tmp_path / "results")
    assert "validation_macro_f1" in per_subject.read_text()
    assert "svm,three_class,0.5" in main.read_text()
