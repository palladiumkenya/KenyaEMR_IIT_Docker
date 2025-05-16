import pandas as pd
import numpy as np
from . import helpers


def prep_visit_features(df):

    # first, parse dob and startartdate with the parse_long_date function
    df['dob'] = df['dob'].apply(helpers.parse_long_date)    
    df['startartdate'] = df['startartdate'].apply(helpers.parse_long_date)

    # replace all cells with "" with None
    df = df.replace(r'^\s*$', None, regex=True)

    # next, apply cleaning functions to the relevant columns
    df['adherence'] = clean_adherence(df['adherence'])
    df['whostage'] = clean_who(df['whostage'])
    df['visittype'] = clean_visittype(df['visittype'])
    df['stabilityassessment'] = clean_stabilityassessment(df['stabilityassessment'])
    df['differentiatedcare'] = clean_differentiatedcare(df['differentiatedcare'])

    # next, generate age from dob and visitdate
    df = gen_age(df)

    # if sex = "male", then set sex = 1, else 0
    df['sex'] = np.where(df['sex'] == 'male', 1, 0)

    # then, clean pregnancy and breastfeeding variables
    df = clean_pregnancy(df)
    df = clean_breastfeeding(df)
    # next, clean bmi variable
    df = clean_bmi(df)
    # finally, clean regimen switch variable
    df = regimen_switch(df)

    # keep only the relevant columns: key, sitecode, visitdate, visittype, visitby,
    #  tcareason, pregnant, pregnant_missing, breastfeeding, breastfeeding_missing,
    #  stabilityassessment, differentiatedcare, whostage, emr, adherence, sex, age,
    #  maritalstatus, educationlevel, occupation, nad_imputed, nad_imputation_flag,
    #  bmi, regimen_switch_visits, startartdate)
    df = df[['key', 'sitecode', 'visitdate', 'visittype', 'visitby',
             'tcareason', 'pregnant', 'pregnant_missing', 'breastfeeding',
             'breastfeeding_missing', 'stabilityassessment', 'differentiatedcare',
             'whostage', 'emr', 'adherence', 'sex', 'age',
             'maritalstatus', 'educationlevel', 'occupation', 'nad_imputed',
             'nad_imputation_flag', 'bmi', 'regimen_switch', 'startartdate']]

    # and return the cleaned dataframe
    return df

def clean_who(who_vec):
    # convert to string
    who_vec = who_vec.astype(str)
    return who_vec

def clean_adherence(adh_vec):
    
    # make lowercase
    adh_vec = adh_vec.str.lower()

    # remove everything after the first pipe character "|"
    adh_vec = adh_vec.str.replace(r'\|.*', '', regex=True)

    # replace values
    adh_vec = adh_vec.str.replace(r'good', '1', regex=True)
    adh_vec = adh_vec.str.replace(r'fair|poor', '0', regex=True)

    # set empty strings to None
    adh_vec = adh_vec.replace(r'^\s*$', None, regex=True)

    # set to None if not '1' or '0'
    adh_vec = adh_vec.where(adh_vec.isin(['1', '0']), None)

    return adh_vec

def clean_visittype(visit_type_vec):

    # make lowercase
    visit_type_vec = visit_type_vec.str.lower()

    # if string contains "unscheduled", then set value to 1, else 0
    visit_type_vec = np.where(visit_type_vec.str.contains('unscheduled', na=False), 1, 0)
    
    return visit_type_vec

def clean_stabilityassessment(stab_assess_vec):
    # make lowercase
    stab_assess_vec = stab_assess_vec.str.lower()

    # Identify blank (empty or whitespace) values
    blank_mask = stab_assess_vec.isna() | (stab_assess_vec.str.strip() == "")

    # if the string contains "un" or "not", set to 0, else 1
    stab_assess_vec = np.where(stab_assess_vec.str.contains('un|not', na=False), 0, 1)

    # Set blanks to None
    stab_assess_vec = np.where(blank_mask, None, stab_assess_vec)

    return stab_assess_vec

def clean_differentiatedcare(diff_care_vec):
    # make lowercase
    diff_care_vec = diff_care_vec.str.lower()

    # group together community art distribution peer led and community art distribution
    # hcw led into a single category of community art distribution
    diff_care_vec = np.where(diff_care_vec.str.contains('community art distribution'), 'community art distribution', diff_care_vec) 

    return diff_care_vec

def gen_age(df):
    # Ensure dob and visitdate are datetime.date
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['visitdate'] = pd.to_datetime(df['visitdate'], errors='coerce')

    # Calculate age in years as a float
    df['age'] = (df['visitdate'] - df['dob']).dt.days / 365.25

    # Remove rows with age less than 0 or where age is missing
    df = df[(df['age'] >= 0) & (df['age'].notnull())]

    return df

def clean_pregnancy(df):
    # define pregnancy_age
    df['pregnancy_age'] = np.where((df['age'] >= 15) & (df['age'] <= 49) & (df['sex'] == 0), 1, 0)

    # Prepare conditions
    yes_mask = (df['pregnancy_age'] == 1) & (df['pregnant'].str.contains('yes', case=False, na=False))
    no_mask = (df['pregnancy_age'] == 1) & (df['pregnant'].str.contains('no', case=False, na=False))

    # Use np.select to assign values
    df['pregnant'] = np.select(
        [yes_mask, no_mask],
        [1, 0],
        default=None
    )

    # finally, create binary flag to indicate if pregnant status is missing or not there because it's not relevant
    df['pregnant_missing'] = np.where(df['pregnant'].isnull() & (df['sex'] == 0) & (df['pregnancy_age'] == 1), 1, 0)

    return df

def clean_breastfeeding(df):
    # Prepare conditions
    yes_mask = (df['pregnancy_age'] == 1) & (df['breastfeeding'].str.contains('yes', case=False, na=False))
    no_mask = (df['pregnancy_age'] == 1) & (df['breastfeeding'].str.contains('no', case=False, na=False))

    # Use np.select to assign values
    df['breastfeeding'] = np.select(
        [yes_mask, no_mask],
        [1, 0],
        default=None
    )

    # Create binary flag to indicate if breastfeeding status is missing or not relevant
    df['breastfeeding_missing'] = np.where(df['breastfeeding'].isnull() & (df['sex'] == 0) & (df['pregnancy_age'] == 1), 1, 0)

    return df

def clean_bmi(df):

    # first, create flags to indicate if the values for height and weight are in a valid range.
    # for height, the valid range is 50 to 250 centimeters
    # for weight, the valid range is 20 to 200 kilograms
    # for each, return 1 to a new variable if the value is in the valid range, else 0
    df['height_valid'] = np.where((df['height'] >= 50) & (df['height'] <= 250), 1, 0)
    df['weight_valid'] = np.where((df['weight'] >= 20) & (df['weight'] <= 200), 1, 0)

    # next, calculate BMI only where the values for height and weight are in a valid range
    # and age is 15 or older
    df['bmi'] = np.where((df['height_valid'] == 1) & (df['weight_valid'] == 1) & (df['age'] >= 15), 
                         df['weight'] / ((df['height'] / 100) ** 2), None)
    
    # finally, group bmi into categories:
    # 1. Underweight: BMI < 18.5    
    # 2. Normalweight: 18.5 <= BMI < 25
    # 3. Overweight: 25 <= BMI < 30
    # 4. Obese: BMI >= 30
    # 5. Missing: BMI is missing
    # 6. if age under 15, then return Under15
    df['bmi'] = np.where(df['age'] < 15, 'Under15', 
                                  np.where(df['bmi'] < 18.5, 'Underweight',
                                           np.where((df['bmi'] >= 18.5) & (df['bmi'] < 25), 'Normalweight',
                                                    np.where((df['bmi'] >= 25) & (df['bmi'] < 30), 'Overweight',
                                                             np.where(df['bmi'] >= 30, 'Obese', None)))))
        
    return df

def regimen_switch(df):
    df = df.copy()
    df['visitdate'] = pd.to_datetime(df['visitdate'], errors='coerce')
    df = df.sort_values(by=['key', 'visitdate'])

    # Treat empty or whitespace-only currentregimen as missing
    df['currentregimen'] = df['currentregimen'].replace(r'^\s*$', None, regex=True)

    # Initialize empty series to collect results
    result_series = pd.Series(index=df.index, dtype="Int64")

    # Group by key
    for key, group in df.groupby('key'):
        group = group.sort_values('visitdate')
        visitdates = group['visitdate']
        regimens = group['currentregimen']
        switch_counts = []

        for i, this_date in enumerate(visitdates):
            mask = (visitdates <= this_date) & (visitdates >= this_date - pd.Timedelta(days=365))
            prev_regimens = regimens[mask].dropna().unique()
            switch_counts.append(len(prev_regimens))

        # Assign to the correct index positions in the original DataFrame
        result_series.loc[group.index] = switch_counts

    # Apply transformation rules
    df['regimen_switch'] = result_series.apply(
        lambda x: None if x == 0 else (0 if x == 1 else 1)
    )
    return df






    

