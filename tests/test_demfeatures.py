import pandas as pd
import numpy as np
from src.common import dem_features

def test_parse_nad_imputed():
    df = pd.DataFrame({'nad_imputed': ['2023-05-12', '2023-05-19', '2023-05-20']})
    df['nad_imputed'] = pd.to_datetime(df['nad_imputed'])
    df = dem_features.parse_nad_imputed(df)
    assert list(df['month']) == [5, 5, 5]
    assert list(df['dayofweek']) == [4, 4, 5]
    assert list(df['is_friday']) == [1, 1, 0]

def test_calculate_daystonextappointment():
    df = pd.DataFrame({
        'visitdate': ['2023-05-10', '2023-05-15'],
        'nad_imputed': ['2023-05-12', '2023-05-20']
    })
    df['visitdate'] = pd.to_datetime(df['visitdate'])
    df['nad_imputed'] = pd.to_datetime(df['nad_imputed'])
    df = dem_features.calculate_daystonextappointment(df)
    assert list(df['daystonextappointment']) == [2, 5]

def test_calculate_timeatfacility():
    df = pd.DataFrame({
        'key': ['A', 'A', 'B'],
        'visitdate': ['2022-01-01', '2022-07-01', '2022-01-01']
    })
    df['visitdate'] = pd.to_datetime(df['visitdate'])
    df = dem_features.calculate_timeatfacility(df)
    # First visit for each key should be 0 months
    assert np.isclose(df.loc[0, 'timeatfacility'], 0, atol=0.1)
    assert np.isclose(df.loc[1, 'timeatfacility'], 6, atol=0.5)
    assert np.isclose(df.loc[2, 'timeatfacility'], 0, atol=0.1)

def test_create_firstvisit_flag():
    df = pd.DataFrame({
        'key': ['A', 'A', 'B'],
        'visitdate': ['2022-01-01', '2022-07-01', '2022-01-01']
    })
    df['visitdate'] = pd.to_datetime(df['visitdate'])
    df = dem_features.create_firstvisit_flag(df)
    assert list(df['firstvisit']) == [1, 0, 1]

def test_calculate_timeonart():
    df = pd.DataFrame({
        'visitdate': ['2022-01-01', '2022-07-01', '2022-01-01'],
        'startartdate': ['2021-01-01', '2021-01-01', '2023-01-01']
    })
    df['visitdate'] = pd.to_datetime(df['visitdate'])
    df['startartdate'] = pd.to_datetime(df['startartdate'])
    df = dem_features.calculate_timeonart(df)
    # First two: 12 and 18 months, last is negative so should be 0
    assert np.isclose(df.loc[0, 'timeonart'], 12, atol=0.5)
    assert np.isclose(df.loc[1, 'timeonart'], 18, atol=0.5)
    assert np.isclose(df.loc[2, 'timeonart'], 0, atol=0.1)

def test_clean_marital_status():
    df = pd.DataFrame({
        'age': [14, 30, 30, 30, 30, 30, 30, 30, 30],
        'maritalstatus': [
            'single', 'single', 'married', 'cohabit', 'divorced',
            'separated', 'widowed', 'poly', 'unknown'
        ]
    })
    df = dem_features.clean_marital_status(df)
    expected = [
        'minor', 'single', 'married', 'married', 'divorced',
        'divorced', 'widowed', 'polygamous', None
    ]
    for val, exp in zip(df['maritalstatus'], expected):
        if exp is None:
            assert pd.isnull(val)
        else:
            assert val == exp

def test_clean_occupation():
    df = pd.DataFrame({
        'occupation': [
            'farmer', 'trader', 'student', 'driver', 'employee',
            'none', 'other', None, 'ml engineer', '  ', 'null'
        ]
    })
    df = dem_features.clean_occupation(df)
    print(df.occupation)
    expected = [
        'farmer', 'trader', 'student', 'driver', 'employee',
        'none', 'other', None, 'other', None, None
    ]
    for val, exp in zip(df['occupation'], expected):
        if exp is None:
            assert pd.isnull(val)
        else:
            assert val == exp

def test_clean_education_level():
    df = pd.DataFrame({
        'educationlevel': [
            'none', 'primary', 'secondary', 'college', 'university', '', None
        ]
    })
    df = dem_features.clean_education_level(df)
    expected = ['none', 'primary', 'secondary', 'college', None, None, None]
    for val, exp in zip(df['educationlevel'], expected):
        if exp is None:
            assert pd.isnull(val)
        else:
            assert val == exp