import pandas as pd
import numpy as np
from datetime import datetime
from . import helpers

def prep_demographics(df):

    # get the month and day of week from the nad_imputed column
    df['month'] = df['nad_imputed'].dt.month
    df['dayofweek'] = df['nad_imputed'].dt.dayofweek
    # create flag is_friday if dayofweek is Friday
    df['is_friday'] = np.where(df['dayofweek'] == 4, 1, 0)

    # create daystonextappointment column as the difference between nad_imputed and visitdate
    df['daystonextappointment'] = (df['nad_imputed'] - df['visitdate']).dt.days

    # calculate timeonart as the difference in months between visitdate and startartdate
    # Ensure visitdate and startartdate are datetime
    df['visitdate'] = pd.to_datetime(df['visitdate'], errors='coerce')
    df['startartdate'] = pd.to_datetime(df['startartdate'], errors='coerce')

    # calculate timeonart as the difference in months between visitdate and startartdate
    df['timeonart'] = (df['visitdate'] - df['startartdate']).dt.days / 30.44
    # replace any negative values with 0
    df['timeonart'] = np.where(df['timeonart'] < 0, 0, df['timeonart'])

    # for each key, get timeatfacility as the difference in months between visitdate and the earliest visitdate
    df['timeatfacility'] = (df['visitdate'] - df.groupby('key')['visitdate'].transform('min')).dt.days / 30.44

    # create a flag called firstvisit if the visitdate is the earliest visitdate for that key
    df['firstvisit'] = np.where(df['visitdate'] == df.groupby('key')['visitdate'].transform('min'), 1, 0)

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

    # now let's clean up occupation as follows
    # first set to lower case
    # categories we'll use are farmer, trader, student, driver,
    # employee, none, other and None
    # if string is farmer, trader, student, driver, employee or none, keep as is
    # if missing, set to None
    # everything else, set as other
    df['occupation'] = df['occupation'].str.lower()
    mask = df['occupation'].isin(['farmer', 'trader', 'student', 'driver', 'employee', 'none'])
    df['occupation'] = np.where(mask, df['occupation'],
                                np.where(df['occupation'] == "null", None, 'other'))
        
    # lastly, let's clean up education level
    # set to lower case
    # if education level is none, primary, secondary, or college, keep as is, otherwise None
    df['educationlevel'] = df['educationlevel'].str.lower()
    df['educationlevel'] = np.where(df['educationlevel'].isin(['none', 'primary', 'secondary', 'college']),
                                     df['educationlevel'], None)
    
    return df
  