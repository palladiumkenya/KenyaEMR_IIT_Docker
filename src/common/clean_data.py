from . import helpers
from datetime import datetime
import pandas as pd


def clean_lab(data, start_date):
    """
    Clean the lab data by removing unnecessary columns and renaming others.
    """

    # if data is empty retun an empty dataframe with the following columns, some of which are
    # not in the data, but we want to return them anyway
    # the columns are:
    # 'key', 'orderedbydate', 'testname', 'testresultcat'
    expected_columns = ["key", "orderedbydate", "testname", "testresultcat"]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )

    # if data is not empty, we proceed with cleaning
    # make column names lower case
    data.columns = data.columns.str.lower()

    # Concatenate patientpkhash and sitecode to create a unique key
    # first make sitecode a string
    data["sitecode"] = data["sitecode"].astype(str)
    # now concatenate
    data["key"] = data["patientpkhash"] + data["sitecode"]

    # parse the date column
    data["orderedbydate"] = data["orderedbydate"].apply(helpers.parse_long_date)

    # Convert start_date to datetime.date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    # Filter the data to only include records after the start_date
    data = data[data["orderedbydate"] >= start_date]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )

    # Let's filter to VLs and CD4
    # First, let's make testname lower case
    # Modify the testname column in place
    data.loc[:, "testname"] = data["testname"].str.lower()

    # Now let's adjust testname so that if it contains the strings % of percent,
    # we consider it a CD4 test; if it contains the string viral or vl, we consider it a VL test
    # and if it contains neither, we consider it an Other test
    # Note: this is a bit of a hack, but it works for now
    # Adjust testname based on the specified conditions
    # Ensure testname has no None or NaN values
    data.loc[:, "testname"] = data["testname"].fillna("")
    data.loc[:, "testname"] = data["testname"].apply(
        lambda x: (
            "CD4"
            if "cd4" in x and "%" not in x and "percent" not in x
            else "VL" if "viral" in x or "vl" in x else "Other"
        )
    )

    # Filter the data to only include CD4 and VL tests
    data = data[data["testname"].isin(["CD4", "VL"])]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )
    # Deduplicate lab data using specialized function
    data = helpers.dedup_lab(data, "key", "orderedbydate", "testname", "testresult")

    return data


def clean_pharmacy(data, start_date, end_date):
    """
    Clean the pharmacy data by removing unnecessary columns and renaming others.
    """

    # if data is empty retun an empty dataframe with the following columns, some of which are
    # not in the data, but we want to return them anyway
    # the columns are:
    # 'key', 'orderedbydate', 'testname', 'testresultcat'
    expected_columns = ["key", "sitecode", "dispensedate", "nad_imputed", "nad_imputation_flag", "drug"]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )

    # make column names lower case
    data.columns = data.columns.str.lower()

    # Concatenate patientpkhash and sitecode to create a unique key
    # first make sitecode a string
    data["sitecode"] = data["sitecode"].astype(str)
    # now concatenate
    data["key"] = data["patientpkhash"] + data["sitecode"]

    # parse the dispensedate and expectedreturn columns
    data["dispensedate"] = data["dispensedate"].apply(helpers.parse_long_date)
    data["expectedreturn"] = data["expectedreturn"].apply(helpers.parse_long_date)

    # Filter data to only include treatmenttype that is either ARV or PMTCT
    data.loc[:, "treatmenttype"] = data["treatmenttype"].str.lower()
    data.loc[:, "treatmenttype"] = data["treatmenttype"].fillna("")
    data = data.loc[data["treatmenttype"].isin(["arv", "pmtct"])]

    # Filter the data to only include records after the start_date
    # Convert start_date to datetime.date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    data = data.loc[data["dispensedate"] >= start_date]
    data = data.loc[data["dispensedate"] <= end_date]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )
    # remove illogical return dates
    data = helpers.remove_date(data, "dispensedate", "expectedreturn")

    # deduplicate pharmacy data using specialized function
    data = helpers.dedup_common(data, "key", "dispensedate", "expectedreturn")

    # impute the expected return date where needed
    data = helpers.impute_date(data, "key", "dispensedate", "expectedreturn")

    # keep the following columns: key, sitecode, dispensedate, nad_imputed, nad_imputation_flag, drug
    data = data[
        [
            "key",
            "sitecode",
            "dispensedate",
            "nad_imputed",
            "nad_imputation_flag",
            "drug",
        ]
    ]

    return data


def clean_visits(data, dem_df, start_date, end_date):
    """
    Clean the visits data by removing unnecessary columns and renaming others.
    """

    # if data is empty retun an empty dataframe with the following columns, some of which are
    # not in the data, but we want to return them anyway
    # the columns are:
    # 'key', 'orderedbydate', 'testname', 'testresultcat'
    expected_columns = ["patientpkhash",
                "sitecode",
                "visitdate",
                "visittype",
                "visitby",
                "nextappointmentdate",
                "tcareason",
                "pregnant",
                "breastfeeding",
                "stabilityassessment",
                "differentiatedcare",
                "whostage",
                "whostagingoi",
                "height",
                "weight",
                "emr",
                "project",
                "adherence",
                "adherencecategory",
                "bp",
                "oi",
                "oidate",
                "currentregimen",
                "appointmentreminderwillingness",
                "key",
                "sex",
                "maritalstatus",
                "educationlevel",
                "occupation",
                "artoutcomedescription",
                "startartdate",
                "dob",
                "nad_imputed",
                "nad_imputation_flag"]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )

    # make column names lower case
    data.columns = data.columns.str.lower()
    dem_df.columns = dem_df.columns.str.lower()

    # first, create key variables
    # Concatenate patientpkhash and sitecode to create a unique key
    # first make sitecode a string
    data["sitecode"] = data["sitecode"].astype(str)
    # now concatenate
    data["key"] = data["patientpkhash"] + data["sitecode"]

    # Repeat for dem_df
    if "key" not in dem_df.columns:
        # sometimes, we have MFL code instead of sitecode, so we need to create sitecode
        if "sitecode" not in dem_df.columns:
            dem_df["sitecode"] = dem_df["mflcode"]
        dem_df["sitecode"] = dem_df["sitecode"].astype(str)
        # now concatenate
        dem_df["key"] = dem_df["patientpkhash"] + dem_df["sitecode"]

    # first, merge visits with demographics (dem) on the "key" column
    data = data.merge(
        dem_df[
            [
                "key",
                "sex",
                "maritalstatus",
                "educationlevel",
                "occupation",
                "artoutcomedescription",
                "startartdate",
                "dob",
            ]
        ],
        on="key",
        how="inner",
    )

    # make the values in each column lower case except for the key column
    cols_to_convert = data.columns.difference(["key"])

    print("DEBUG: data.shape before applymap:", data.shape)
    print("DEBUG: cols_to_convert:", cols_to_convert)
    print("DEBUG: data[cols_to_convert].shape:", data[cols_to_convert].shape)
    print("DEBUG: data[cols_to_convert].head():\n", data[cols_to_convert].head())

    print("DEBUG: data.columns:", list(data.columns))
    print("DEBUG: Are there duplicates?", len(data.columns) != len(set(data.columns)))
    print("DEBUG: END")

    data[cols_to_convert] = data[cols_to_convert].applymap(
        lambda x: x.lower() if isinstance(x, str) else x
    )

    # parse the visitdate column
    data["visitdate"] = data["visitdate"].apply(helpers.parse_long_date)
    data["nextappointmentdate"] = data["nextappointmentdate"].apply(
        helpers.parse_long_date
    )

    # Filter the data to only include records after the start_date
    # Convert start_date to datetime.date
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    data = data[data["visitdate"] >= start_date]
    data = data[data["visitdate"] <= end_date]
    if data.empty:
        return pd.DataFrame(
            columns=expected_columns
        )
    # remove whitespace from all columns
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # remove illogical return dates
    data = helpers.remove_date(data, "visitdate", "nextappointmentdate")

    # deduplicate visit data using specialized function
    data = helpers.dedup_common(data, "key", "visitdate", "nextappointmentdate")

    # impute the expected return date where needed
    data = helpers.impute_date(data, "key", "visitdate", "nextappointmentdate")

    return data
