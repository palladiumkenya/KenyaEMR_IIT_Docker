import pandas as pd
import numpy as np

def create_target(visits_df, pharmacy_df, dem_df):

    """
    Create the target variable for the model.

    Args:
        visits_df (pd.DataFrame): DataFrame containing visit data.
        pharmacy_df (pd.DataFrame): DataFrame containing pharmacy data.
        dem_df (pd.DataFrame): DataFrame containing demographic data.

    Returns:
        pd.DataFrame: DataFrame with the target variable added.
    """
    
    # select the relevant columns from visits_df and rename nad_imputed to nad
    # relevant columns are key, visitdate, nad_imputed, nad_imputation_flag, sitecode
    visits_df = visits_df[['key', 'visitdate', 'nad_imputed', 'nad_imputation_flag', 'sitecode']]
    # rename nad_imputed to nad
    visits_df = visits_df.rename(columns={'nad_imputed': 'nad'})
    # create a new variably called type and set it to 'clinial'
    visits_df['type'] = 'clinical'

    # select the relevant columns from pharmacy_df and rename nad_imputed to nad 
    # rename dispense_date to visitdate
    # relevant columns are key, dispensedate, nad_imputed, nad_imputation_flag, sitecode
    pharmacy_df = pharmacy_df[['key', 'dispensedate', 'nad_imputed', 'nad_imputation_flag', 'sitecode']]
    # rename nad_imputed to nad
    pharmacy_df = pharmacy_df.rename(columns={'nad_imputed': 'nad'})
    # rename dispense_date to visitdate
    pharmacy_df = pharmacy_df.rename(columns={'dispensedate': 'visitdate'})
    # create a new variably called type and set it to 'pharmacy'
    pharmacy_df['type'] = 'pharmacy'
 
    # Now, vertically stack the two dataframes
    # concatenate the two dataframes    
    target_df = pd.concat([visits_df, pharmacy_df], axis=0)

    # take dem_df, set variables to lower case, and select key and artoutcomedescription
    dem_df.columns = dem_df.columns.str.lower()
    dem_df = dem_df[['key', 'artoutcomedescription']]
    # set the values in the artoutcomedescription column to lower case and remove whitespace
    dem_df.loc[:, 'artoutcomedescription'] = (dem_df.loc[:, 'artoutcomedescription']
                                              .str.lower()
    )

    # merge target_df and dem_df on key
    target_df = target_df.merge(dem_df, on='key', how='left')

    ## Deduplicate clinical and pharmacy data
    #  if there are multiple rows with same key / visitdate, one from clinical and one from pharmacy:
    # 1. prioritize nad_imputation_flag of 0 (non-imputed)
    # 2. within that, keep the row with later nad date
    # to do this, first sort the dataframe so that for each key and visitdate, nad_imputation flag of 0
    # is first, then nad_imputation_flag of 1, and then within each group of nad_imputation flag, sort by
    # nad is descending order so that the later nad date is first.
    target_df['visitdate'] = pd.to_datetime(target_df['visitdate'], errors='coerce')
    # if there are more than one row per key and visitdate, set type to 'both'
    target_df['type'] = np.where(
        target_df.duplicated(subset=['key', 'visitdate'], keep=False),
        'both',
        target_df['type']
    )
    target_df = target_df.sort_values(by=['key', 'visitdate', 'nad_imputation_flag', 'nad'],
                                       ascending=[True, True, True, False])
    # print(target_df.shape)
    # now, group by key and visitdate and take the first row 
    target_df = target_df.groupby(['key', 'visitdate']).first().reset_index()
    # drop rows where type is 'pharmacy'
    target_df = target_df[target_df['type'] != 'pharmacy']


    ## Deal with out of order NAD 
    # sort the dataframe by key and visitdate in descending order
    target_df = target_df.sort_values(by=['key', 'visitdate'], ascending=[True, True])
    # create variable nad2 which for each row is going to be the max nad observed
    # over all earlier touchpoints for that key, meaning the rows below
    target_df['nad2'] = target_df.groupby('key')['nad'].cummax()

    # create nad_imputation_flag_ooo column
    # if nad2 is not equal to nad, set nad_imputation_flag_ooo to 1, else 0 
    target_df['nad_imputation_flag_ooo'] = np.where(
        target_df['nad2'] != target_df['nad'], 1, 0
    )
    # update imputation flag - if either nad_imputation_flag or nad_imputation_flag_ooo is 1, set nad_imputation_flag to 1
    target_df['nad_imputation_flag'] = np.where(
        (target_df['nad_imputation_flag'] == 1) | (target_df['nad_imputation_flag_ooo'] == 1), 1, 0
    )
    # set nad to nad2 and drop nad2 and nad_imputation_flag_ooo
    target_df['nad'] = target_df['nad2']
    target_df = target_df.drop(columns=['nad2', 'nad_imputation_flag_ooo'])

    ## Calculate days to return and iit
    # for each key, sort by visitdate in descending order
    target_df = target_df.sort_values(by=['key', 'visitdate'], ascending=[True, False])
    # create a variable called num_visit which is the row number of the visit for each key
    target_df['num_visit'] = target_df.groupby('key').cumcount()
    # create 'actualreturndate' as the visitdate from the row above (ie the next visit)
    target_df['actualreturndate'] = target_df.groupby('key')['visitdate'].shift(1)
    # create a variable called visitdiff which is the difference between actualreturndate and nad
    target_df['visitdiff'] = (target_df['actualreturndate'] - target_df['nad']).dt.days
    # create a variable called iit which is 1 if visitdiff is greater than 30, else 0
    target_df['iit'] = np.where(target_df['visitdiff'] > 30, 1, 0)

    ## Account for most recent visit
    # if the not enough time has passed to observe an outcome for the most recent visit,
    # then drop the row. we will determine this by checking if the facility reported 
    # any visits more than 30 days after the nad. if they did, then we'll say the outcome
    # is iit, but if they didn't, then we'll say it's unresolved and drop the row.
    # first, split the dataframe into two dataframes - one for the most recent visit and one for all other visits
    most_recent_visit = target_df[target_df['num_visit'] == 0]
    other_visits = target_df[target_df['num_visit'] > 0]

    # get the max visitdate for each sitecode
    max_visitdate = target_df.groupby('sitecode')['visitdate'].max().reset_index()

    # merge most_recent_visit with max_visitdate on sitecode
    most_recent_visit = most_recent_visit.merge(max_visitdate, on='sitecode', how='left', suffixes=('', '_max'))

    # for each row, create a variable called outcome
    # if the nad + 30 days is greater than the max visitdate, then set outcome to
    # unresolved, else set it to iit
    most_recent_visit['outcome'] = np.where(
        (most_recent_visit['nad'] + pd.Timedelta(days=30)) > most_recent_visit['visitdate_max'],
        'unresolved',
        'iit'
    )

    # drop 'unresolved' rows from most_recent_visit
    most_recent_visit = most_recent_visit[most_recent_visit['outcome'] == 'iit']
    # set iit to 1
    most_recent_visit['iit'] = 1
    # drop the visitdate_max column
    most_recent_visit = most_recent_visit.drop(columns=['visitdate_max', 'outcome'])

    # now, filter most_recent_visit to only include patients who did not die
    # or have a documented transfer out, since those would not be considered IIT.
    # filter to artoucomedescription of "active", "losstofollowup", "lostinhmis"    
    most_recent_visit = most_recent_visit[
        most_recent_visit['artoutcomedescription'].isin(['active', 'loss to follow up', 'lost in hmis'])
    ]

    # now, vertically stack the two dataframes
    # concatenate the two dataframes
    target_df = pd.concat([other_visits, most_recent_visit], axis=0)
    # drop the num_visit and actualreturndate columns
    target_df = target_df.drop(columns=['num_visit', 'artoutcomedescription', 'type'])

    return target_df