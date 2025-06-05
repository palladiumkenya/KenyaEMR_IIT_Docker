import pandas as pd
import numpy as np
from src.common import visit_features


def test_clean_adherence():
    s = pd.Series(["Good|Fair", "Fair", "Poor", "", None])
    result = visit_features.clean_adherence(s)
    assert list(result) == ["1", "0", "0", None, None]


def test_clean_visittype():
    s = pd.Series(["Unscheduled", "Routine", "UNSCHEDULED follow-up", None])
    result = visit_features.clean_visittype(s)
    assert list(result) == [1, 0, 1, 0]


def test_clean_stabilityassessment():
    s = pd.Series(["Unstable", "Not stable", "Stable", "", None])
    result = visit_features.clean_stabilityassessment(s)
    assert list(result) == [0, 0, 1, None, None]


def test_clean_differentiatedcare():
    s = pd.Series(["Community ART Distribution Peer Led", "HCW Led", "Other"])
    result = visit_features.clean_differentiatedcare(s)
    assert list(result) == ["community art distribution", "hcw led", "other"]


def test_gen_age():
    df = pd.DataFrame(
        {"dob": ["2000-01-01", "2010-01-01"], "visitdate": ["2020-01-01", "2020-01-01"]}
    )
    df = visit_features.gen_age(df)
    # Age should be about 20 and 10
    assert np.allclose(df["age"], [20, 10], atol=0.1)


def test_clean_pregnancy():
    df = pd.DataFrame(
        {"age": [25, 25, 10], "sex": [0, 0, 0], "pregnant": ["yes", "no", "yes"]}
    )
    df = visit_features.clean_pregnancy(df)
    assert list(df["pregnant"]) == [1, 0, None]
    assert list(df["pregnant_missing"]) == [0, 0, 0]


def test_clean_breastfeeding():
    df = pd.DataFrame(
        {
            "pregnancy_age": [1, 1, 0],
            "sex": [0, 0, 0],
            "breastfeeding": ["yes", "no", "yes"],
        }
    )
    df = visit_features.clean_breastfeeding(df)
    assert list(df["breastfeeding"]) == [1, 0, None]
    assert list(df["breastfeeding_missing"]) == [0, 0, 0]


def test_clean_bmi():
    df = pd.DataFrame(
        {"height": [170, 160, 100], "weight": [70, 40, 15], "age": [20, 20, 10]}
    )
    df = visit_features.clean_bmi(df)
    assert list(df["bmi"]) == ["Normalweight", "Underweight", "Under15"]


def test_regimen_switch():
    df = pd.DataFrame(
        {
            "key": ["A", "A", "A", "B", "B"],
            "visitdate": [
                "2020-01-01",
                "2020-06-01",
                "2020-10-01",
                "2020-04-01",
                "2021-01-01",
            ],
            "currentregimen": ["ABC", "ABC", "DEF", "XYZ", "XYZ"],
        }
    )
    df = visit_features.regimen_switch(df)
    # For patient A: first two visits should be None, third should be 1 (switch)
    # For patient B: first visit None, second visit 0 (no switch)
    assert list(df["regimen_switch"]) == [0, 0, 1, 0, 0]
