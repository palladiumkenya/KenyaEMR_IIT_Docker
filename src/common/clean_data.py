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
