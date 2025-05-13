import pandas as pd
import numpy as np

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

    return targets_df
