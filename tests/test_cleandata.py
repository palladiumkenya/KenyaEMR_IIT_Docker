import pandas as pd
from src.common.clean_data import clean_lab


def test_no_missing_values_in_testname_or_testresult():
    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash2"],
            "sitecode": [123, 456],
            "orderedbydate": ["2020-01-02", "2020-01-03"],
            "testname": ["cd4 test", "cd4 test"],
            "testresult": ["result1", None],
        }
    )

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert no missing values in testname or testresult
    assert cleaned_data["testname"].notnull().all(), "Missing values found in testname"
    assert (
        cleaned_data["testresultcat"].notnull().all()
    ), "Missing values found in testresult"


def test_only_allowed_values_in_testname():
    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash2", "hash3"],
            "sitecode": [123, 456, 789],
            "orderedbydate": ["2020-01-02", "2020-01-03", "2020-01-04"],
            "testname": ["cd4 test", "viral load", "unknown test"],
            "testresult": ["result1", "result2", "result3"],
        }
    )

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert only allowed values are in testname
    allowed_values = {"CD4", "VL"}
    assert set(cleaned_data["testname"].unique()).issubset(
        allowed_values
    ), "Invalid values found in testname"


def test_no_duplicate_entries():
    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1"],
            "sitecode": [123, 123],
            "orderedbydate": ["2020-01-02", "2020-01-02"],
            "testname": ["cd4 test", "cd4 test"],
            "testresult": ["result1", "result1"],
        }
    )

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert no duplicate entries based on orderedbydate and testname
    assert not cleaned_data.duplicated(
        subset=["orderedbydate", "testname"]
    ).any(), "Duplicate entries found"


def test_clean_pharmacy_treatmenttype_filter():
    from src.common.clean_data import clean_pharmacy

    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1", "hash3"],
            "sitecode": [123, 123, 456],
            "treatmenttype": ["ARV", "PMTCT", "Other"],
            "drug": ["DrugA", "DrugB", "DrugC"],
            "dispensedate": ["2022-10-01", "2023-01-01", "2023-02-01"],
            "expectedreturn": ["2023-01-01", "2023-02-01", "2023-03-01"],
        }
    )

    # Clean the data
    cleaned_data = clean_pharmacy(data, start_date="2020-01-01", end_date="2025-01-01")

    # Assert only rows with allowed treatment types are retained
    assert len(cleaned_data) == 2, "Rows with invalid treatment types were not filtered"


def test_clean_pharmacy_date_parsing():
    from src.common.clean_data import clean_pharmacy

    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1"],
            "sitecode": [123, 123],
            "treatmenttype": ["ARV", "PMTCT"],
            "drug": ["DrugA", "DrugB"],
            "dispensedate": ["2022-10-01", "2023-01-01"],
            "expectedreturn": ["2023-01-01T12:00:00", None],
        }
    )

    # Clean the data
    cleaned_data = clean_pharmacy(data, start_date="2020-01-01", end_date="2025-01-01")

    # Assert dates are parsed correctly
    assert (
        pd.to_datetime(cleaned_data["nad_imputed"], errors="coerce").notnull().all()
    ), "Dates not parsed correctly"


def test_clean_pharmacy_date_range_filter():
    from src.common.clean_data import clean_pharmacy

    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1"],
            "sitecode": [123, 123],
            "treatmenttype": ["ARV", "PMTCT"],
            "drug": ["DrugA", "DrugB"],
            "dispensedate": ["2019-12-31", "2023-01-01"],
            "expectedreturn": ["2020-12-31", "2023-01-01"],
        }
    )

    # Clean the data
    cleaned_data = clean_pharmacy(data, start_date="2020-01-01", end_date="2025-01-01")

    # Assert only rows within the date range are retained
    assert len(cleaned_data) == 1, "Rows outside the date range were not filtered"


def test_clean_visits_date_parsing():
    from src.common.clean_data import clean_visits

    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1"],
            "sitecode": [123, 123],
            "visitdate": ["2023-01-01T12:00:00", None],
            "nextappointmentdate": ["2023-02-01", "invalid-date"],
        }
    )

    # add dem_df to match the function signature with the following columns
    #  "key","sex", "maritalstatus", "educationlevel", "occupation",
    # "artoutcomedescription", "startartdate", "dob"
    dem_df = pd.DataFrame(
        {
            "key": ["hash1" + "123"],
            "sex": ["M"],
            "maritalstatus": ["Single"],
            "educationlevel": ["High School"],
            "occupation": ["Engineer"],
            "artoutcomedescription": ["On ART"],
            "startartdate": ["2020-01-01"],
            "dob": ["1990-01-01"],
        }
    )

    # Clean the data
    cleaned_data = clean_visits(
        data, dem_df=dem_df, start_date="2020-01-01", end_date="2025-01-01"
    )

    # Assert dates are parsed correctly
    assert (
        pd.to_datetime(cleaned_data["visitdate"], errors="coerce").notnull().all()
    ), "visitdate not parsed correctly"
    assert (
        pd.to_datetime(cleaned_data["nextappointmentdate"], errors="coerce")
        .notnull()
        .all()
    ), "nextappointmentdate not parsed correctly"


def test_clean_visits_date_range_filter():
    from src.common.clean_data import clean_visits

    # Mock input data
    data = pd.DataFrame(
        {
            "patientpkhash": ["hash1", "hash1"],
            "sitecode": [123, 123],
            "visitdate": ["2019-12-31", "2023-01-01"],
            "nextappointmentdate": ["2023-02-01", "2023-03-01"],
        }
    )

    dem_df = pd.DataFrame(
        {
            "key": ["hash1" + "123"],
            "sex": ["M"],
            "maritalstatus": ["Single"],
            "educationlevel": ["High School"],
            "occupation": ["Engineer"],
            "artoutcomedescription": ["On ART"],
            "startartdate": ["2020-01-01"],
            "dob": ["1990-01-01"],
        }
    )

    # Clean the data
    cleaned_data = clean_visits(
        data, dem_df=dem_df, start_date="2020-01-01", end_date="2025-01-01"
    )
    print(cleaned_data)
    # Assert only rows within the date range are retained
    assert len(cleaned_data) == 1, "Rows outside the date range were not filtered"


def test_parse_long_date():
    from src.common.helpers import parse_long_date
    from datetime import datetime

    # Test valid date
    assert (
        parse_long_date("2023-01-01T12:00:00") == datetime(2023, 1, 1).date()
    ), "Valid date not parsed correctly"

    # Test invalid date
    assert parse_long_date("invalid-date") is None, "Invalid date not handled correctly"

    # Test None
    assert parse_long_date(None) is None, "None value not handled correctly"
