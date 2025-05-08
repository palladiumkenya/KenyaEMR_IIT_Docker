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
    # Take the first 10 characters and parse them into a date
    return datetime.strptime(date_col[:10], "%Y-%m-%d").date()

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
        elif row[labname_var] == "CD4":
            if row['testresultnum'] < 200:
                return "NoAHD"
            elif row['testresultnum'] >= 200:
                return "YesAHD"
        return None

    df['testresultcat'] = df.apply(classify, axis=1)

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