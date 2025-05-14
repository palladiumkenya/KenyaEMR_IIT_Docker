import pandas as pd
import numpy as np
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
    targets_df = targets_df.sort_values(by=['key', 'visitdate'])
    # Create date_reengaged column. 
    # if iit = 1 at the previous visit, then date_reengaged = visitdate
    # if iit = 0, then date_reenaged is None
    targets_df['visitdate'] = pd.to_datetime(targets_df['visitdate'], errors='coerce')
    targets_df['iit_lag'] = targets_df.groupby('key')['iit'].shift(1)
    targets_df['date_reengaged'] = targets_df.apply(
        lambda x: x['visitdate'] if x['iit_lag'] == 1 else None, axis=1
    )
    # Fill forward the date_reengaged column
    targets_df['date_reengaged'] = targets_df.groupby('key')['date_reengaged'].ffill()
    targets_df['date_reengaged'] = pd.to_datetime(targets_df['date_reengaged'], errors='coerce')
    # Calculate the time to reengagement as visitdate - date_reengaged      
    targets_df['monthssincerestart'] = (targets_df['visitdate'] - targets_df['date_reengaged']).dt.days / 30

    # Categorize monthssincerestart into bins: 
    # if None, then "neverdisengaged"
    # if 0-6 months, "shorttermrestart"
    # if >6 months, then "longtermrestart"
    targets_df['monthssincerestart'] = targets_df['monthssincerestart'].fillna(-1)
    targets_df['cascadestatus'] = targets_df['monthssincerestart'].apply(
        lambda x: 'neverdisengaged' if x == -1 else 
        ('shorttermrestart' if x <= 6 else 'longtermrestart')
    )

    # drop monthssincerestart, date_reengaged, and iit_lag columns
    targets_df = targets_df.drop(columns=['monthssincerestart', 'date_reengaged', 'iit_lag'])

    ## Rolling join with visits_df
    # Merge the targets DataFrame with the visits DataFrame on 'key' and 'visitdate'
    # We want a rolling join, with the visit just before the target visit
    targets_df['join_time'] = targets_df['visitdate'] #- pd.Timedelta('1ns')
    visits_df['join_time'] = visits_df['visitdate']
    visits_df = visits_df.drop(columns=['visitdate', 'sitecode'])

    targets_df['join_time'] = pd.to_datetime(targets_df['join_time'])
    visits_df['join_time'] = pd.to_datetime(visits_df['join_time'])
    targets_df['key'] = targets_df['key'].astype(str)
    visits_df['key'] = visits_df['key'].astype(str)
    targets_df = targets_df.sort_values(['key', 'join_time'], ascending=[True, True]).reset_index(drop=True)
    visits_df = visits_df.sort_values(['key', 'join_time'], ascending=[True, True]).reset_index(drop=True)

    targets_df = pl.from_pandas(targets_df)
    visits_df = pl.from_pandas(visits_df)

    targets_df = targets_df.join_asof(
        visits_df,
        left_on="join_time",
        right_on="join_time",
        by="key",             # join by group
        strategy="backward"       # or 'forward' or 'nearest'
    ).to_pandas()
    
    # targets_df = pd.merge_asof(
    #     targets_df,
    #     visits_df,
    #     by='key',
    #     on='join_time',
    #     direction='backward'  
    # )

    ## Lateness metrics
    # first, clean up visitdiff. if visitdiff is less than 0, then set to 0.
    #  if over 100, set to 100
    #  if None, keep as None
    targets_df['visitdiff'] = targets_df['visitdiff'].clip(lower=0, upper=100)

    # create lastvd column as visitdiff from the previous visit
    targets_df['lastvd'] = targets_df.groupby('key')['visitdiff'].shift(1)

    # create three binaries. 
    # if lastvd is greater than 0, then late = 1, else late = 0
    # if lastvd is greater than 14, then late14 = 1, else late14 = 0
    # if lastvd is greater than 30, then late30 = 1, else late30 = 0
    targets_df['late'] = targets_df['lastvd'].apply(lambda x: 1 if x > 0 else 0)
    targets_df['late14'] = targets_df['lastvd'].apply(lambda x: 1 if x > 14 else 0)
    targets_df['late30'] = targets_df['lastvd'].apply(lambda x: 1 if x > 30 else 0)

    # now, let's create rolling features
    # first, let's get the rolling mean of visitdiff over the last 3, 5, and 10 visits
    targets_df['lateness_last3'] = targets_df.groupby('key')['lastvd'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    targets_df['lateness_last5'] = targets_df.groupby('key')['lastvd'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    targets_df['lateness_last10'] = targets_df.groupby('key')['lastvd'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    # now, let's get the rolling sum of late over the last 3, 5, and 10 visits
    targets_df['late_last3'] = targets_df.groupby('key')['late'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df['late_last5'] = targets_df.groupby('key')['late'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df['late_last10'] = targets_df.groupby('key')['late'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    # now, let's get the rolling sum of late14 over the last 3, 5, and 10 visits
    targets_df['late14_last3'] = targets_df.groupby('key')['late14'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df['late14_last5'] = targets_df.groupby('key')['late14'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df['late14_last10'] = targets_df.groupby('key')['late14'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )   
    # now, let's get the rolling sum of late30 over the last 3, 5, and 10 visits
    targets_df['late30_last3'] = targets_df.groupby('key')['late30'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )
    targets_df['late30_last5'] = targets_df.groupby('key')['late30'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    targets_df['late30_last10'] = targets_df.groupby('key')['late30'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )

    return targets_df
