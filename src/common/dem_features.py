import pandas as pd
import numpy as np
from datetime import datetime
from . import helpers

def prep_demographics(df):

    # make sure the columns are in the right format
    df['visitdate'] = pd.to_datetime(df['visitdate'], errors='coerce')
    df['startartdate'] = pd.to_datetime(df['startartdate'], errors='coerce')
    df['nad_imputed'] = pd.to_datetime(df['nad_imputed'], errors='coerce')

    # get the month and day of week from the nad_imputed column
    df = parse_nad_imputed(df)
    # calculate daystonextappointment as the difference in days between nad_imputed and visitdate
    df = calculate_daystonextappointment(df)
    # calculate timeonart as the difference in months between visitdate and startartdate
    df = calculate_timeonart(df)
    # calculate timeatfacility as the difference in months between visitdate and the earliest visitdate for each key
    df = calculate_timeatfacility(df)
    # create a flag called firstvisit if the visitdate is the earliest visitdate for that key
    df = create_firstvisit_flag(df) 
    # clean marital status, occupation and education level
    df = clean_marital_status(df)
    df = clean_occupation(df)
    df = clean_education_level(df)

    return df

def clean_marital_status(df):

    # clean up the maritalstatus column as follows
    # set to lower case. then, if age is under 15, set to "minor"
    # if maritalstatus contains the string single, set to "single"
    # if maritalstatus contains the string married or cohabit, set to "married"
    # if maritalstatus contains the string divorced or separated, set to "divorced"
    # if maritalstatus contains the string widowed, set to "widowed"
    # if maritalstatus contains the string poly, set to "polygamous"
    # if none of these apply, set to None
    df['maritalstatus'] = df['maritalstatus'].str.lower()
    df['maritalstatus'] = np.where(df['age'] < 15, 'minor', df['maritalstatus'])
    df['maritalstatus'] = np.where(df['maritalstatus'].str.contains('poly'), 'polygamous', df['maritalstatus'])
    df['maritalstatus'] = np.where(df['maritalstatus'].str.contains('single'), 'single', df['maritalstatus'])
    df['maritalstatus'] = np.where(df['maritalstatus'].str.contains('married|cohabit'), 'married', df['maritalstatus'])
    df['maritalstatus'] = np.where(df['maritalstatus'].str.contains('divorced|separated'), 'divorced', df['maritalstatus'])
    df['maritalstatus'] = np.where(df['maritalstatus'].str.contains('widowed'), 'widowed', df['maritalstatus'])
    df['maritalstatus'] = np.where(
        ~df['maritalstatus'].isin(['minor', 'single', 'married', 'divorced', 'widowed', 'polygamous']),
        None,
        df['maritalstatus']
    )

    return df

def clean_occupation(df):
    # set to lower case
    df['occupation'] = df['occupation'].str.lower()

    # replace whitespace-only or empty strings with None
    df['occupation'] = df['occupation'].replace(r'^\s*$', None, regex=True)

    allowed = ['farmer', 'trader', 'student', 'driver', 'employee', 'none', 'other']
    df['occupation'] = df['occupation'].apply(
        lambda x: x if x in allowed else (None if x == "null" or x is None else "other")
    )

    return df

def clean_education_level(df):

    # clean up the education level as follows
    # set to lower case. if education level is none, primary, secondary, or college, keep as is, otherwise None
    df['educationlevel'] = df['educationlevel'].str.lower()
    df['educationlevel'] = np.where(df['educationlevel'].isin(['none', 'primary', 'secondary', 'college']),
                                     df['educationlevel'], None)
    
    return df

def parse_nad_imputed(df):
    # Extract the month and day of the week from the 'nad_imputed' column
    df['month'] = df['nad_imputed'].dt.month
    df['dayofweek'] = df['nad_imputed'].dt.dayofweek

    # Create a flag for Fridays
    df['is_friday'] = np.where(df['dayofweek'] == 4, 1, 0)

    return df

def calculate_daystonextappointment(df):
    
    # Calculate daystonextappointment as the difference in days between nad_imputed and visitdate
    df['daystonextappointment'] = (df['nad_imputed'] - df['visitdate']).dt.days

    return df

def calculate_timeatfacility(df):

    # Calculate timeatfacility as the difference in months between visitdate and the earliest visitdate for each key
    df['timeatfacility'] = (df['visitdate'] - df.groupby('key')['visitdate'].transform('min')).dt.days / 30.44

    return df

def create_firstvisit_flag(df):
    # Create a flag called firstvisit if the visitdate is the earliest visitdate for that key
    df['firstvisit'] = np.where(df['visitdate'] == df.groupby('key')['visitdate'].transform('min'), 1, 0)

    return df

def calculate_timeonart(df):

    # Calculate timeonart as the difference in months between visitdate and startartdate
    df['timeonart'] = (df['visitdate'] - df['startartdate']).dt.days / 30.44

    # Replace any negative values with 0
    df['timeonart'] = np.where(df['timeonart'] < 0, 0, df['timeonart'])

    return df