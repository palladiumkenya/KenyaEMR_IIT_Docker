from . import helpers
from datetime import datetime

def clean_lab(data, start_date):
    """
    Clean the lab data by removing unnecessary columns and renaming others.
    """
    # make column names lower case
    data.columns = data.columns.str.lower()

    # Concatenate patientpkhash and sitecode to create a unique key
    # first make sitecode a string
    data['sitecode'] = data['sitecode'].astype(str)
    # now concatenate
    data['key'] = data['patientpkhash'] + data['sitecode']
  
    # parse the date column
    data['orderedbydate'] = data['orderedbydate'].apply(helpers.parse_long_date)

    # Convert start_date to datetime.date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    # Filter the data to only include records after the start_date
    data = data[data['orderedbydate'] >= start_date]

    # Let's filter to VLs and CD4
    # First, let's make testname lower case
    # Modify the testname column in place   
    data.loc[:, 'testname'] = data['testname'].str.lower()
    
    # Now let's adjust testname so that if it contains the strings % of percent,
    # we consider it a CD4 test; if it contains the string viral or vl, we consider it a VL test
    # and if it contains neither, we consider it an Other test
    # Note: this is a bit of a hack, but it works for now
    # Adjust testname based on the specified conditions
    # Ensure testname has no None or NaN values
    data.loc[:, 'testname'] = data['testname'].fillna('')
    data.loc[:, 'testname'] = data['testname'].apply(
        lambda x: 'CD4' if 'cd4' in x and '%' not in x and 'percent' not in x else
                'VL' if 'viral' in x or 'vl' in x else
                'Other'
    )

    # Filter the data to only include CD4 and VL tests
    data = data[data['testname'].isin(['CD4', 'VL'])]

    # Deduplicate lab data using specialized function
    data = helpers.dedup_lab(data, 'key', 'orderedbydate', 'testname', 'testresult')
    
    return data

def clean_pharmacy(data, start_date, end_date):
    """
    Clean the pharmacy data by removing unnecessary columns and renaming others.
    """
    # make column names lower case
    data.columns = data.columns.str.lower()

    # Concatenate patientpkhash and sitecode to create a unique key
    # first make sitecode a string
    data['sitecode'] = data['sitecode'].astype(str)
    # now concatenate
    data['key'] = data['patientpkhash'] + data['sitecode']

    # parse the dispensedate and expectedreturn columns
    data['dispensedate'] = data['dispensedate'].apply(helpers.parse_long_date)
    data['expectedreturn'] = data['expectedreturn'].apply(helpers.parse_long_date)

    # Filter data to only include treatmenttype that is either ARV or PMTCT 
    data.loc[:, 'treatmenttype'] = data['treatmenttype'].str.lower()
    data.loc[:, 'treatmenttype'] = data['treatmenttype'].fillna('')
    data = data.loc[data['treatmenttype'].isin(['arv', 'pmtct'])]

    # Filter the data to only include records after the start_date
    # Convert start_date to datetime.date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    data = data.loc[data['dispensedate'] >= start_date]
    data = data.loc[data['dispensedate'] <= end_date]

    # remove illogical return dates
    data = helpers.remove_date(data, 'dispensedate', 'expectedreturn')

    # deduplicate pharmacy data using specialized function
    data = helpers.dedup_common(data, 'key', 'dispensedate', 'expectedreturn')

    # impute the expected return date where needed
    data = helpers.impute_date(data, 'key', 'dispensedate', 'expectedreturn')

    # keep the following columns: key, sitecode, dispensedate, nad_imputed, nad_imputation_flag, drug
    data = data[['key', 'sitecode', 'dispensedate', 'nad_imputed', 'nad_imputation_flag', 'drug']]

    return data