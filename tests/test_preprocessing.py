"""Session 3: preprocessing code test (Module 1 Titanic cleaning)."""
import pandas as pd

from ml.preprocessing import clean_titanic


def test_no_missing_age():
    sample = pd.DataFrame({"Age": [22, None, 35], "Embarked": ["S", "C", "Q"]})
    cleaned = clean_titanic(sample)
    assert cleaned["Age"].isnull().sum() == 0


def test_drops_missing_embarked():
    sample = pd.DataFrame({"Age": [22, 30], "Embarked": ["S", None]})
    cleaned = clean_titanic(sample)
    assert len(cleaned) == 1
