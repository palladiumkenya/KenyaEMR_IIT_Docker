import pandas as pd
from src.common.clean_data import clean_lab

def test_no_missing_values_in_testname_or_testresult():
    # Mock input data
    data = pd.DataFrame({
        'patientpkhash': ['hash1', 'hash2'],
        'sitecode': [123, 456],
        'orderedbydate': ['2020-01-02', '2020-01-03'],
        'testname': ['cd4 test', 'cd4 test'],
        'testresult': ['result1', None]
    })

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert no missing values in testname or testresult
    assert cleaned_data['testname'].notnull().all(), "Missing values found in testname"
    assert cleaned_data['testresultcat'].notnull().all(), "Missing values found in testresult"


def test_only_allowed_values_in_testname():
    # Mock input data
    data = pd.DataFrame({
        'patientpkhash': ['hash1', 'hash2', 'hash3'],
        'sitecode': [123, 456, 789],
        'orderedbydate': ['2020-01-02', '2020-01-03', '2020-01-04'],
        'testname': ['cd4 test', 'viral load', 'unknown test'],
        'testresult': ['result1', 'result2', 'result3']
    })

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert only allowed values are in testname
    allowed_values = {'CD4', 'VL'}
    assert set(cleaned_data['testname'].unique()).issubset(allowed_values), "Invalid values found in testname"


def test_no_duplicate_entries():
    # Mock input data
    data = pd.DataFrame({
        'patientpkhash': ['hash1', 'hash1'],
        'sitecode': [123, 123],
        'orderedbydate': ['2020-01-02', '2020-01-02'],
        'testname': ['cd4 test', 'cd4 test'],
        'testresult': ['result1', 'result1']
    })

    # Clean the data
    cleaned_data = clean_lab(data, start_date="2020-01-01")

    # Assert no duplicate entries based on orderedbydate and testname
    assert not cleaned_data.duplicated(subset=['orderedbydate', 'testname']).any(), "Duplicate entries found"

