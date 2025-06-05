import pandas as pd
import polars as pl


def prep_target_visit_features(targets_df, visits_df):
    """
    Prepares target visit features by merging the targets DataFrame with the visits DataFrame.

    Parameters:
    - targets_df (pd.DataFrame): The DataFrame containing target data.
    - visits_df (pd.DataFrame): The DataFrame containing visit data.

    Returns:
    - pd.DataFrame: A DataFrame containing the merged target visit features.
    """
    ## Cascade features
    # First, for every instance of IIT, we want to calculate how long until reengagement
    # sort by key and in ascending order of visitdate
    targets_df = targets_df.sort_values(by=["key", "visitdate"])
    # Create date_reengaged column.
    # if iit = 1 at the previous visit, then date_reengaged = visitdate
    # if iit = 0, then date_reenaged is None
    targets_df["visitdate"] = pd.to_datetime(targets_df["visitdate"], errors="coerce")
    targets_df["iit_lag"] = targets_df.groupby("key")["iit"].shift(1)
    targets_df["date_reengaged"] = targets_df.apply(
        lambda x: x["visitdate"] if x["iit_lag"] == 1 else None, axis=1
    )
    # Fill forward the date_reengaged column
    targets_df["date_reengaged"] = targets_df.groupby("key")["date_reengaged"].ffill()
    targets_df["date_reengaged"] = pd.to_datetime(
        targets_df["date_reengaged"], errors="coerce"
    )
    # Calculate the time to reengagement as visitdate - date_reengaged
    targets_df["monthssincerestart"] = (
        targets_df["visitdate"] - targets_df["date_reengaged"]
    ).dt.days / 30

    # Categorize monthssincerestart into bins:
    # if None, then "neverdisengaged"
    # if 0-6 months, "shorttermrestart"
    # if >6 months, then "longtermrestart"
    targets_df["monthssincerestart"] = targets_df["monthssincerestart"].fillna(-1)
    targets_df["cascadestatus"] = targets_df["monthssincerestart"].apply(
        lambda x: (
            "neverdisengaged"
            if x == -1
            else ("shorttermrestart" if x <= 6 else "longtermrestart")
        )
    )

    # drop monthssincerestart, date_reengaged, and iit_lag columns
    targets_df = targets_df.drop(
        columns=["monthssincerestart", "date_reengaged", "iit_lag"]
    )

    ## Rolling join with visits_df
    # Merge the targets DataFrame with the visits DataFrame on 'key' and 'visitdate'
    # We want a rolling join, with the visit just before the target visit
    targets_df["join_time"] = targets_df["visitdate"]  # - pd.Timedelta('1ns')
    visits_df["join_time"] = visits_df["visitdate"]
    visits_df = visits_df.drop(
        columns=["visitdate", "sitecode", "nad_imputation_flag", "nad_imputed"]
    )

    targets_df["join_time"] = pd.to_datetime(targets_df["join_time"])
    visits_df["join_time"] = pd.to_datetime(visits_df["join_time"])
    targets_df["key"] = targets_df["key"].astype(str)
    visits_df["key"] = visits_df["key"].astype(str)
    targets_df = targets_df.sort_values(
        ["key", "join_time"], ascending=[True, True]
    ).reset_index(drop=True)
    visits_df = visits_df.sort_values(
        ["key", "join_time"], ascending=[True, True]
    ).reset_index(drop=True)

    targets_df = pl.from_pandas(targets_df)
    visits_df = pl.from_pandas(visits_df)

    targets_df = targets_df.join_asof(
        visits_df,
        left_on="join_time",
        right_on="join_time",
        by="key",  # join by group
        strategy="backward",  # or 'forward' or 'nearest'
    ).to_pandas()

    ## Lateness metrics
    # first, clean up visitdiff. if visitdiff is less than 0, then set to 0.
    #  if over 100, set to 100
    #  if None, keep as None
    targets_df["visitdiff"] = targets_df["visitdiff"].clip(lower=0, upper=100)

    # create lastvd column as visitdiff from the previous visit
    targets_df["lastvd"] = targets_df.groupby("key")["visitdiff"].shift(1)
    targets_df = targets_df.drop(columns=["join_time", "visitdiff"])
    # create three binaries.
    # if lastvd is greater than 0, then late = 1, else late = 0
    # if lastvd is greater than 14, then late14 = 1, else late14 = 0
    # if lastvd is greater than 30, then late30 = 1, else late30 = 0
    targets_df["late"] = targets_df["lastvd"].apply(lambda x: 1 if x > 0 else 0)
    targets_df["late14"] = targets_df["lastvd"].apply(lambda x: 1 if x > 14 else 0)
    targets_df["late30"] = targets_df["lastvd"].apply(lambda x: 1 if x > 30 else 0)

    # now, let's create rolling features
    # first, let's get the rolling mean of visitdiff over the last 3, 5, and 10 visits
    targets_df["lateness_last3"] = targets_df.groupby("key")["lastvd"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    targets_df["lateness_last5"] = targets_df.groupby("key")["lastvd"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    targets_df["lateness_last10"] = targets_df.groupby("key")["lastvd"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    # now, let's get the rolling sum of late over the last 3, 5, and 10 visits
    targets_df["late_last3"] = targets_df.groupby("key")["late"].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df["late_last5"] = targets_df.groupby("key")["late"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df["late_last10"] = targets_df.groupby("key")["late"].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    # now, let's get the rolling sum of late14 over the last 3, 5, and 10 visits
    targets_df["late14_last3"] = targets_df.groupby("key")["late14"].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df["late14_last5"] = targets_df.groupby("key")["late14"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df["late14_last10"] = targets_df.groupby("key")["late14"].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    # now, let's get the rolling sum of late30 over the last 3, 5, and 10 visits
    targets_df["late30_last3"] = targets_df.groupby("key")["late30"].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df["late30_last5"] = targets_df.groupby("key")["late30"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df["late30_last10"] = targets_df.groupby("key")["late30"].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )

    return targets_df


def prep_target_pharmacy_features(targets_df, pharmacy_df):
    """
    Prepares target pharmacy features by merging the targets DataFrame with the pharmacy DataFrame.

    Parameters:
    - targets_df (pd.DataFrame): The DataFrame containing target data.
    - pharmacy_df (pd.DataFrame): The DataFrame containing pharmacy data.

    Returns:
    - pd.DataFrame: A DataFrame containing the merged target pharmacy features.
    """
    # first, create a new column in pharmacy_df called optimizedregimen that is 1
    # if the drug variable contains the string "DTG", else 0
    pharmacy_df["optimizedhivregimen"] = pharmacy_df["drug"].apply(
        lambda x: 1 if "DTG" in x else 0
    )

    # select key, visitdate in place of dispensedate, and optimizedregimen
    pharmacy_df["visitdate"] = pharmacy_df["dispensedate"]
    pharmacy_df["visitdate"] = pd.to_datetime(pharmacy_df["visitdate"], errors="coerce")
    pharmacy_df = pharmacy_df[["key", "visitdate", "optimizedhivregimen"]]

    pharmacy_df.loc[:, "key"] = pharmacy_df.loc[:, "key"].astype(str)

    # do rolling join with targets_df
    targets_df = targets_df.sort_values(
        ["key", "visitdate"], ascending=[True, True]
    ).reset_index(drop=True)
    pharmacy_df = pharmacy_df.sort_values(
        ["key", "visitdate"], ascending=[True, True]
    ).reset_index(drop=True)

    targets_df = pl.from_pandas(targets_df)
    pharmacy_df = pl.from_pandas(pharmacy_df)

    targets_df = targets_df.join_asof(
        pharmacy_df,
        left_on="visitdate",
        right_on="visitdate",
        by="key",  # join by group
        strategy="backward",  # or 'forward' or 'nearest'
    ).to_pandas()

    return targets_df


def prep_target_lab_features(targets_df, lab_df):
    """
    Prepares target lab features by merging the targets DataFrame with the lab DataFrame.

    Parameters:
    - targets_df (pd.DataFrame): The DataFrame containing target data.
    - lab_df (pd.DataFrame): The DataFrame containing lab data.

    Returns:
    - pd.DataFrame: A DataFrame containing the merged target lab features.
    """
    # we'll need to join vl and cd4 data separately onto targets_df since they
    # can be taken on different days
    # first, get vl data
    vl_df = lab_df[lab_df["testname"] == "VL"]
    # rename testrestultcat to vl
    vl_df.loc[:, "vl"] = vl_df.loc[:, "testresultcat"]
    vl_df = vl_df[["key", "orderedbydate", "vl"]]

    targets_df["visitdate"] = pd.to_datetime(targets_df["visitdate"])
    vl_df["orderedbydate"] = pd.to_datetime(vl_df["orderedbydate"])

    # do rolling join with targets_df
    targets_df = targets_df.sort_values(
        ["key", "visitdate"], ascending=[True, True]
    ).reset_index(drop=True)
    vl_df = vl_df.sort_values(
        ["key", "orderedbydate"], ascending=[True, True]
    ).reset_index(drop=True)

    targets_df = pl.from_pandas(targets_df)
    vl_df = pl.from_pandas(vl_df)

    targets_df = (
        targets_df.join_asof(
            vl_df,
            left_on="visitdate",
            right_on="orderedbydate",
            by="key",
            strategy="backward",
        )
        .with_columns(
            (pl.col("visitdate") - pl.col("orderedbydate"))
            .dt.total_days()
            .alias("days_diff")
        )
        .with_columns(
            pl.when((pl.col("days_diff") > 365) | (pl.col("days_diff") < 0))
            .then(None)
            .otherwise(pl.col("vl"))
            .alias("most_recent_vl")
        )
        .drop("days_diff", "orderedbydate", "vl")
        .to_pandas()
    )

    # where most_recent_vl is None, if timeonart is less than 6 months,
    # then set to "earlyart". Otherwise, if most_recent_vl is none,
    # time on art is greater than six months but timeatfacility is less than
    # 6 months, then set to "restart". finally, any remaining missing
    # most_recent_vl should be set to "novalidvl".
    def classify_vl(row):
        if pd.isnull(row["most_recent_vl"]):
            if pd.notnull(row["timeonart"]) and row["timeonart"] <= 6:
                return "earlyart"
            elif (
                pd.notnull(row["timeonart"])
                and row["timeonart"] > 6
                and pd.notnull(row["timeatfacility"])
                and row["timeatfacility"] <= 6
            ):
                return "restart"
            else:
                return "novalidvl"
        else:
            return row["most_recent_vl"]

    targets_df["most_recent_vl"] = targets_df.apply(classify_vl, axis=1)

    # now, let's repeat the process for cd4 data
    cd4_df = lab_df[lab_df["testname"] == "CD4"]
    # rename testrestultcat to cd4
    cd4_df.loc[:, "cd4"] = cd4_df.loc[:, "testresultcat"]
    cd4_df = cd4_df[["key", "orderedbydate", "cd4"]]
    cd4_df["orderedbydate"] = pd.to_datetime(cd4_df["orderedbydate"], errors="coerce")
    # do rolling join with targets_df
    targets_df = targets_df.sort_values(
        ["key", "visitdate"], ascending=[True, True]
    ).reset_index(drop=True)
    cd4_df = cd4_df.sort_values(
        ["key", "orderedbydate"], ascending=[True, True]
    ).reset_index(drop=True)
    targets_df = pl.from_pandas(targets_df)
    cd4_df = pl.from_pandas(cd4_df)

    targets_df = (
        targets_df.join_asof(
            cd4_df,
            left_on="visitdate",
            right_on="orderedbydate",
            by="key",  # join by group
            strategy="backward",  # or 'forward' or 'nearest'
        )
        .with_columns(
            (pl.col("visitdate") - pl.col("orderedbydate"))
            .dt.total_days()
            .alias("days_diff")
        )
        .with_columns(
            pl.when((pl.col("days_diff") > 365) | (pl.col("days_diff") < 0))
            .then(None)
            .otherwise(pl.col("cd4"))
            .alias("most_recent_cd4")
        )
        .drop("days_diff", "orderedbydate")
        .to_pandas()
    )

    # finally, create a variable called ahd.
    # if age is less than 5 or cd4 is "YesAHD" or whostage is 3 or 4, then ahd = 1
    # else ahd = 0
    targets_df["ahd"] = targets_df.apply(
        lambda x: (
            1
            if (x["age"] < 5)
            or (x["most_recent_cd4"] == "YesAHD")
            or (x["whostage"] in [3, 4])
            else 0
        ),
        axis=1,
    )

    # drop most_recent_cd4 column
    targets_df = targets_df.drop(columns=["most_recent_cd4", "cd4"])

    return targets_df
