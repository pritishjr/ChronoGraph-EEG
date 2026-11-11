from chronograph.smoke import run_smoke


def test_synthetic_smoke_runs_one_epoch():
    result = run_smoke()
    assert result["epochs"] == 1
    assert result["samples"] == 6
    assert 0.0 <= result["svm_validation_macro_f1"] <= 1.0
    assert 0.0 <= result["slate_validation_macro_f1"] <= 1.0
