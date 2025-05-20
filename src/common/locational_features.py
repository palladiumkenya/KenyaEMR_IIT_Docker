import pandas as pd
import numpy as np


def prep_locational_features(targets_df, mfl, dhs, txcurr):
    """
    This function adds locational features to the targets DataFrame by merging it with MFL, DHS, and TXCURR data.

    Parameters:
    targets_df (pd.DataFrame): The DataFrame containing target data.
    mfl (pd.DataFrame): The DataFrame containing MFL data.
    dhs (pd.DataFrame): The DataFrame containing DHS data.
    txcurr (pd.DataFrame): The DataFrame containing TXCURR data.

    Returns:
    pd.DataFrame: The updated targets DataFrame with locational features.
    """
    
    # set MFL column names to lower case
    mfl.columns = mfl.columns.str.lower()

    # remove periods and underscores from column names
    mfl.columns = mfl.columns.str.replace('.', '', regex=False)   
    mfl.columns = mfl.columns.str.replace('_', '', regex=False)

    # select code, kephlevel, county, facilitytypecategory, and ownertype
    mfl = mfl[['code', 'kephlevel', 'county', 'facilitytypecategory', 'ownertype']]
    
    # set DHS column names to lower case and remove periods and underscores
    dhs.columns = dhs.columns.str.lower()
    dhs.columns = dhs.columns.str.replace('.', '', regex=False)

    # delect country and survey columns
    dhs = dhs.drop(columns=['country', 'survey'], errors='ignore')

    # merge MFL and DHS by county
    mfl_dhs = pd.merge(mfl, dhs, how='left', left_on='county', right_on='county')

    # merge mfl_dhs with targets_df, by sitecode for targets_df and code for mfl_dhs
    # first, make sure sitecode and code are the same type
    targets_df['sitecode'] = targets_df['sitecode'].astype(str)
    mfl_dhs['code'] = mfl_dhs['code'].astype(str)
    targets_df = pd.merge(targets_df, mfl_dhs, how='left', left_on='sitecode', right_on='code')

    # Now, txcurr. First, set txcurr column names to lower case and remove periods
    txcurr.columns = txcurr.columns.str.lower()
    # rename facilitycode to sitecode and indicator_value to txcurr 
    txcurr = txcurr.rename(columns={'facilitycode': 'sitecode', 'indicator_value': 'txcurr'})

    # Ensure targets_df['date'] is datetime
    targets_df['visitdate'] = pd.to_datetime(targets_df['visitdate'])

    # Create period column in YYYYMM format as integer
    targets_df['period'] = targets_df['visitdate'].dt.strftime('%Y%m').astype(int)

    # Ensure txcurr['period'] is integer
    txcurr['period'] = txcurr['period'].astype(int)

    # make sure sitecode and period are the same type in both
    targets_df['sitecode'] = targets_df['sitecode'].astype(str)
    txcurr['sitecode'] = txcurr['sitecode'].astype(str)
    # select sitecode, period, and txcurr
    # Merge on sitecode and period
    targets_df = pd.merge(
        targets_df,
        txcurr[['sitecode', 'period', 'txcurr']],
        how='left',
        on=['sitecode', 'period']
    )

    # now let's take targets_df and for each sitecode and each month,
    # get the rolling weighted no show rate for the last 6 months
    # and rolling weighted days late for the last 6 months.
    # first, create a new column called last_iit which is 1 if
    # lastvd is greater than 30, else 0
    # now, create a new dataframe in which we take targets_df and
    #  groupby sitecode and period and get the mean of last_iit,
    # get the mean of lastvd (removing NA if necessary),
    #  and get the count of observations. we can keep only 
    # sitecode, period, last_iit, lastvd and count to save memory
    df = targets_df[['sitecode', 'period', 'lastvd']]
    df['last_iit'] = (df['lastvd'] > 30).astype(int)
    df = df.groupby(['sitecode', 'period']).agg(
        last_iit=('last_iit', 'mean'),
        lastvd=('lastvd', 'mean'),
        count=('lastvd', 'count')
    ).reset_index()

    # now, groupby sitecode and sort by period and get the weighted 
    # rolling mean of last_iit and lastvd over the last 6 months
    # Weighted rolling mean function
    def weighted_rolling_mean(values, weights, window):
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            v = values[start:i+1]
            w = weights[start:i+1]
            if np.sum(w) == 0:
                result.append(np.nan)
            else:
                result.append(np.sum(v * w) / np.sum(w))
        return np.array(result)

    df = df.sort_values(['sitecode', 'period'])

    # Apply weighted rolling mean for last_iit and lastvd
    df['rolling_weighted_noshow'] = (
        df.groupby('sitecode')
        .apply(lambda g: weighted_rolling_mean(g['last_iit'].values, g['count'].values, 6))
        .explode()
        .astype(float)
        .values
    )
    df['rolling_weighted_dayslate'] = (
        df.groupby('sitecode')
        .apply(lambda g: weighted_rolling_mean(g['lastvd'].values, g['count'].values, 6))
        .explode()
        .astype(float)
        .values
    )

    # before merging, update period column in df to be one period earlier
    # Convert period to string, then to datetime (first day of month)
    df['period_dt'] = pd.to_datetime(df['period'].astype(str) + '01', format='%Y%m%d')

    # Subtract one month
    df['period_dt'] = df['period_dt'] - pd.DateOffset(months=1)

    # Convert back to YYYYMM integer
    df['period'] = df['period_dt'].dt.strftime('%Y%m').astype(int)

    # Drop the helper column
    df = df.drop(columns=['period_dt', 'last_iit'])

    # Now, merge these back to targets_df on sitecode and period
    targets_df = pd.merge(
        targets_df,
        df[['sitecode', 'period', 'rolling_weighted_noshow', 'rolling_weighted_dayslate']],
        how='left',
        on=['sitecode', 'period']
    )

    targets_df = targets_df.drop(columns = ['period'])
  
    return targets_df