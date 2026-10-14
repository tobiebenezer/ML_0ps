"""Classroom Titanic cleaning fn (Session 3 preprocessing test target)."""
import pandas as pd


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df = df.dropna(subset=["Embarked"])
    return df
