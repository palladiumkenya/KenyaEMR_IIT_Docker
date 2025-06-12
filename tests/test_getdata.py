import sqlite3
import pytest
import os
from src.common.get_data import get_training_data
from src.common.get_data import get_inference_data

# test cases for get_training_data function


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Database not available in CI")
def test_get_data_no_prediction():
    # Call the function with prediction=False
    lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_training_data(aws=False)

    # Assert that the returned DataFrames are not empty
    assert not lab.empty, "Lab DataFrame is empty"
    assert not pharmacy.empty, "Pharmacy DataFrame is empty"
    assert not visits.empty, "Visits DataFrame is empty"
    assert not dem.empty, "Demographics DataFrame is empty"
    assert not mfl.empty, "MFL DataFrame is empty"
    assert not dhs.empty, "DHS DataFrame is empty"
    assert not txcurr.empty, "Txcurr DataFrame is empty"


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Database not available in CI")
def test_get_data_with_prediction_valid():
    # Mock valid patientPK and sitecode
    patientPK = "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F"
    sitecode = "13074"

    # Call the function with prediction=True
    lab, pharmacy, visits, dem = get_inference_data(
        patientPK=patientPK, sitecode=sitecode
    )

    # Assert that the returned DataFrames are not empty
    assert not lab.empty, "Lab DataFrame is empty for valid patientPK and sitecode"
    assert (
        not pharmacy.empty
    ), "Pharmacy DataFrame is empty for valid patientPK and sitecode"
    assert (
        not visits.empty
    ), "Visits DataFrame is empty for valid patientPK and sitecode"
    assert (
        not dem.empty
    ), "Demographics DataFrame is empty for valid patientPK and sitecode"


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Database not available in CI")
def test_get_data_with_prediction_invalid():
    # Mock invalid patientPK and sitecode
    patientPK = "invalid_patient_pk"
    sitecode = "invalid_site_code"

    # Call the function with prediction=True
    lab, pharmacy, visits, dem = get_inference_data(
        patientPK=patientPK, sitecode=sitecode
    )

    # Assert that the returned DataFrames are empty
    assert lab.empty, "Lab DataFrame is not empty for invalid patientPK and sitecode"
    assert (
        pharmacy.empty
    ), "Pharmacy DataFrame is not empty for invalid patientPK and sitecode"
    assert (
        visits.empty
    ), "Visits DataFrame is not empty for invalid patientPK and sitecode"
    assert (
        dem.empty
    ), "Demographics DataFrame is not empty for invalid patientPK and sitecode"


def test_get_data_database_error(monkeypatch):
    # Use monkeypatch to simulate a missing database file
    monkeypatch.setattr(
        "sqlite3.connect",
        lambda _: (_ for _ in ()).throw(sqlite3.OperationalError("Database not found")),
    )

    with pytest.raises(sqlite3.OperationalError, match="Database not found"):
        get_training_data(aws=False)
