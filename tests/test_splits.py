from chronograph.splits.leakage_checks import assert_no_leakage
from chronograph.splits.loso import build_folds


def test_loso_uses_subject_test_and_seizure_validation_groups():
    records = [
        {"id": "a1", "subject": "a", "seizure_id": "a:s1", "label": "ictal"},
        {"id": "a2", "subject": "a", "seizure_id": "a:s2", "label": "preictal"},
        {"id": "b1", "subject": "b", "seizure_id": "b:s1", "label": "ictal"},
        {"id": "b2", "subject": "b", "seizure_id": "b:s2", "label": "interictal"},
        {"id": "c1", "subject": "c", "seizure_id": "c:s1", "label": "ictal"},
        {"id": "c2", "subject": "c", "seizure_id": "c:s2", "label": "preictal"},
    ]
    fold = build_folds(records, validation_fraction=0.5, seed=42)[0]
    assert set(fold["train"] + fold["validation"]) == {"b1", "b2", "c1", "c2"}
    assert fold["test"] == ["a1", "a2"]
    assert_no_leakage(records, fold["train"], fold["validation"], fold["test"])


def test_loso_keeps_alias_subjects_in_one_fold():
    records = [
        {"id": "a1", "subject": "chb01", "seizure_id": "a:s1", "label": "ictal"},
        {"id": "b1", "subject": "chb21", "seizure_id": "b:s1", "label": "ictal"},
        {"id": "c1", "subject": "chb02", "seizure_id": "c:s1", "label": "ictal"},
    ]
    folds = build_folds(records)
    patient_fold = next(fold for fold in folds if fold["fold"] == "patient_01")
    assert set(patient_fold["test"]) == {"a1", "b1"}
