import pandas as pd
from src.common import target_features


def test_prep_target_visit_features_basic():
    # Minimal input with two visits for one patient
    targets = pd.DataFrame(
        {
            "key": ["A", "A", "A"],
            "visitdate": ["2022-04-01", "2022-10-01", "2023-04-01"],
            "iit": [0, 1, 0],
            "visitdiff": [15, 60, 0],
            "sitecode": ["001", "001", "001"],
            "nad_imputation_flag": [0, 0, 0],
            "nad_imputed": ["2022-09-15", "2023-02-01", "2023-07-01"],
        }
    )
    visits = pd.DataFrame(
        {
            "key": ["A", "A", "A"],
            "visitdate": ["2021-10-01", "2022-04-01", "2022-10-01"],
            "sitecode": ["001", "001", "001"],
            "nad_imputation_flag": [0, 0, 0],
            "nad_imputed": ["2022-01-01", "2022-09-15", "2023-04-01"],
        }
    )
    targets["visitdate"] = pd.to_datetime(targets["visitdate"])
    targets["nad_imputed"] = pd.to_datetime(targets["nad_imputed"])
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])

    out = target_features.prep_target_visit_features(targets.copy(), visits.copy())
    # Check that rolling features exist
    assert "lateness_last3" in out.columns
    assert "late_last3" in out.columns
    # Check that cascadestatus is correct
    assert out.loc[0, "cascadestatus"] == "neverdisengaged"
    assert out.loc[1, "cascadestatus"] in [
        "shorttermrestart",
        "longtermrestart",
        "neverdisengaged",
    ]
    assert out.loc[2, "cascadestatus"] == "shorttermrestart"
    # Check that late/late14/late30 are correct
    assert out.loc[1, "late"] == 1
    assert out.loc[1, "late14"] == 1
    assert out.loc[1, "late30"] == 0


# def test_prep_target_visit_features_single_row():
#     # Single visit should not error and should fill with defaults
#     targets = pd.DataFrame({
#         "key": ["A"],
#         "visitdate": ["2022-01-01"],
#         "iit": [0],
#         "visitdiff": [0],
#         "sitecode": ["001"],
#         "nad_imputation_flag": [0],
#         "nad_imputed": ["2022-01-01"]
#     })
#     visits = targets.copy()
#     targets["visitdate"] = pd.to_datetime(targets["visitdate"])
#     targets["nad_imputed"] = pd.to_datetime(targets["nad_imputed"])
#     visits["visitdate"] = pd.to_datetime(visits["visitdate"])
#     visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
#     out = target_features.prep_target_visit_features(targets.copy(), visits.copy())
#     assert "lateness_last3" in out.columns
#     assert out.shape[0] == 1


def test_prep_target_pharmacy_features_basic():
    targets = pd.DataFrame({"key": ["A"], "visitdate": ["2022-01-01"]})
    pharmacy = pd.DataFrame(
        {"key": ["A"], "dispensedate": ["2022-01-01"], "drug": ["DTG"]}
    )
    targets["visitdate"] = pd.to_datetime(targets["visitdate"])
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    out = target_features.prep_target_pharmacy_features(targets.copy(), pharmacy.copy())
    assert "optimizedhivregimen" in out.columns
    assert out["optimizedhivregimen"].iloc[0] == 1


def test_prep_target_lab_features_vl_cd4_logic():
    targets = pd.DataFrame(
        {
            "key": ["A"],
            "visitdate": ["2022-01-01"],
            "timeonart": [2],
            "timeatfacility": [2],
            "age": [30],
            "whostage": [1],
        }
    )
    lab = pd.DataFrame(
        {
            "key": ["A", "A"],
            "testname": ["VL", "CD4"],
            "orderedbydate": ["2021-12-01", "2021-12-01"],
            "testresultcat": ["suppressed", "YesAHD"],
        }
    )
    targets["visitdate"] = pd.to_datetime(targets["visitdate"])
    lab["orderedbydate"] = pd.to_datetime(lab["orderedbydate"])
    out = target_features.prep_target_lab_features(targets.copy(), lab.copy())
    print(out["most_recent_vl"])
    # Should fill most_recent_vl with "suppressed" because VL is 500
    assert "most_recent_vl" in out.columns
    assert out["most_recent_vl"].iloc[0] == "suppressed"
    # Should fill ahd as 1 because most_recent_cd4 is YesAHD
    assert "ahd" in out.columns
    assert out["ahd"].iloc[0] == 1


# def test_prep_target_lab_features_missing_vl_cd4():
#     # No lab data: should fill with novalidvl and ahd=0
#     targets = pd.DataFrame({
#         "key": ["A"],
#         "visitdate": ["2022-01-01"],
#         "timeonart": [10],
#         "timeatfacility": [10],
#         "age": [30],
#         "whostage": [1]
#     })
#     lab = pd.DataFrame(columns=["key", "testname", "orderedbydate", "testresultcat"])
#     targets["visitdate"] = pd.to_datetime(targets["visitdate"])
#     out = target_features.prep_target_lab_features(targets.copy(), lab.copy())
#     assert out["most_recent_vl"].iloc[0] == "novalidvl"
#     assert out["ahd"].iloc[0] == 0
