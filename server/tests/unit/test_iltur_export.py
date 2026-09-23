import pytest

from app.ingest.iltur_export import assert_disjoint


def test_disjoint_splits_pass():
    assert_disjoint({"train": ["1", "2"], "dev": ["3"], "test": ["4"]})


def test_test_case_leaking_into_train_is_rejected():
    with pytest.raises(AssertionError, match="both 'train' and 'test'"):
        assert_disjoint({"train": ["1", "2"], "dev": ["3"], "test": ["2"]})


def test_duplicate_within_one_split_is_allowed():
    assert_disjoint({"train": ["1", "1"], "test": ["9"]})
