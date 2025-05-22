from datetime import datetime
import pandas as pd
import numpy as np

def parse_long_date(date_col):
    """
    Parse a long string into a date object.

    Args:
        date_col (str): The date string to parse.

    Returns:
        datetime.date: The parsed date object.
    """
    try:
        # Convert to string and take the first 10 characters
        return datetime.strptime(str(date_col)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        # Return None if parsing fails
        return None

def remove_date(df, contact_var, return_var):
    """
    Remove the date from a contact variable.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        contact_var (str): The name of the contact date variable.
        return_var (str): The name of the expected return date variable.

    Returns:
        pd.DataFrame: The cleaned contact variable without the date.
    """
    # make sure contact_var and return_var are date types
    df[contact_var] = pd.to_datetime(df[contact_var], errors='coerce')
    df[return_var] = pd.to_datetime(df[return_var], errors='coerce')

    # create a new column that is the number of days between the contact date and the return date
    df['days_between'] = (df[return_var] - df[contact_var]).dt.days

    # if the number of days is less than 0 or more than 365, set the return var to None
    df.loc[df['days_between'] <= 0, return_var] = None
    df.loc[df['days_between'] > 365, return_var] = None

    # drop the days_between column
    df.drop(columns=['days_between'], inplace=True)

    # return the dataframe
    return df  

def dedup_common(df, key_var, contact_var, return_var):
    """
    Deduplicate common data by removing duplicates based on key variables.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        key_var (str): The name of the key variable.
        contact_var (str): The name of the contact date variable.
        return_var (str): The name of the expected return date variable.

    Returns:
        pd.DataFrame: The deduplicated DataFrame.
    """
    # Group by key_ver and contact_var, sort in descending order by return_var,
    # and keep the first occurrence with no ties
    df = df.sort_values(by=[key_var, contact_var, return_var], ascending=[True, True, False])

    # Drop duplicates, keeping the first occurrence (top entry after sorting)
    df = df.drop_duplicates(subset=[key_var, contact_var], keep='first')

    return df

def impute_date(df, key_var, contact_var, return_var):
    """
    Impute missing dates in the contact variable based on the gap 
    between the return variable and contact variable set at the prior contact date.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        key_var (str): The name of the key variable.
        contact_var (str): The name of the contact date variable.
        return_var (str): The name of the expected return date variable.

    Returns:
        pd.DataFrame: The DataFrame with imputed dates.
    """
    # Convert contact_var and return_var to datetime
    df[contact_var] = pd.to_datetime(df[contact_var], errors='coerce')
    df[return_var] = pd.to_datetime(df[return_var], errors='coerce')

    # Get the number of days between the contact and return date
    df['days_between'] = (df[return_var] - df[contact_var]).dt.days

    # Create a days_between_group variable using the following logic:
    # If days_between is less than 45, then set it to 30
    # If days_between is greater than 45 and less than 75, then set it to 60
    # If days_between is greater than 75 and less than 105, then set it to 90
    # If days_between is greater than 105 and less than 150, then set it to 120
    # If days_between is greater than 150 and less than 200, then set it to 180
    # Else, set it to 30 
    df['days_between_group'] = np.select(
        [
            df['days_between'] < 45,
            (df['days_between'] >= 45) & (df['days_between'] < 75),
            (df['days_between'] >= 75) & (df['days_between'] < 105),
            (df['days_between'] >= 105) & (df['days_between'] < 150),
            (df['days_between'] >= 150) & (df['days_between'] < 200)
        ],
        [
            30,
            60,
            90,
            120,
            180
        ],
        default=30
    )

    # Sort by key_var and contact_var to ensure proper ordering
    df = df.sort_values(by=[key_var, contact_var], ascending=[True, False])

    # Group by key_var and apply the imputation logic
    def impute_group(group):
        # Carry forward the days_between_group from the prior row
        group['nad_imputed'] = group.apply(
            lambda row: row[contact_var] + pd.Timedelta(days=group.loc[group.index[group.index.get_loc(row.name) - 1], 'days_between_group'])
            if pd.isna(row[return_var]) else row[return_var],
            axis=1
        )
        return group

    df = df.groupby(key_var, group_keys=False).apply(impute_group).reset_index(drop =True)

    # If any nad_imputed is still missing, impute it as contact_var plus 30 days
    df['nad_imputed'] = df.apply(
        lambda row: row[contact_var] + pd.Timedelta(days=30) if pd.isna(row['nad_imputed']) else row['nad_imputed'],
        axis=1
    )

    # set a nad_imputation_flag variable to 1 if nad_imputed is not equal to nad, else 0
    df['nad_imputation_flag'] = np.where(df['nad_imputed'] != df[return_var], 1, 0)

    # drop the days_between and days_between_group columns
    df.drop(columns=['days_between', 'days_between_group'], inplace=True)

    # return the dataframe`
    return df

def dedup_lab(df, key_var, contact_var, labname_var, labresult_var):
    # Group by key variables and count labs
    df['lab_count'] = df.groupby([key_var, contact_var, labname_var])[labname_var].transform('count')

    # Create is_multi variable
    df['is_multi'] = np.where(df['lab_count'] > 1, 1, 0)

    # Clean and parse lab results
    df['testresultraw'] = df[labresult_var]
    df['testresultclean'] = df['testresultraw'].astype(str).str.replace('.', '', regex=False).str.lower()
    df['testresultnum'] = pd.to_numeric(df['testresultraw'], errors='coerce')

    # Create testresultcat
    def classify(row):
        if row[labname_var] == "VL":
            if pd.isna(row['testresultnum']) or row['testresultnum'] < 200:
                return "suppressed"
            elif row['testresultnum'] >= 200:
                return "nonsuppressed"
        # for CD4, if the testresultnum is NaN, return None because
        # it's never a string that can be converted to a number
        # and we don't want to keep it
        elif pd.isna(row['testresultnum']):
            return None
        # if the testresultnum is not NaN, we can classify it
        elif row[labname_var] == "CD4":
            if row['testresultnum'] < 200:
                return "NoAHD"
            elif row['testresultnum'] >= 200:
                return "YesAHD"
        return None

    df['testresultcat'] = df.apply(classify, axis=1)

    # Drop rows from CD4 that could not be classified
    df = df[~((df[labname_var] == "CD4") & (df['testresultnum'].isna()))]

    # Split into single and multi
    df_single = df[df['is_multi'] == 0]
    df_multi = df[df['is_multi'] == 1]

    # Clean multi where only one non-NA numeric result
    df_multi_grouped = df_multi.copy()
    df_multi_grouped['num_result_count'] = df_multi_grouped.groupby([key_var, contact_var, labname_var])['testresultnum'].transform(lambda x: x.notna().sum())
    df_multi_clean = df_multi_grouped[
        (df_multi_grouped['num_result_count'] == 1) & (df_multi_grouped['testresultnum'].notna())
    ]

    df_cleaned = pd.concat([df_single, df_multi_clean], ignore_index=True)

    # Multi where all results agree
    df_multi_grouped['num_result_count'] = df_multi_grouped.groupby([key_var, contact_var, labname_var])['testresultnum'].transform(lambda x: x.notna().sum())
    df_multi_grouped['num_cats'] = df_multi_grouped.groupby([key_var, contact_var, labname_var])['testresultcat'].transform(lambda x: x.nunique())
    df_multi_agree = df_multi_grouped[
        (df_multi_grouped['num_result_count'] > 1) & (df_multi_grouped['num_cats'] == 1)
    ]
    print(len(df_multi_agree))

    # Combine all cleaned
    df_final = pd.concat([df_cleaned, df_multi_agree], ignore_index=True)

    return df_final[[key_var, contact_var, labname_var, 'testresultcat']]