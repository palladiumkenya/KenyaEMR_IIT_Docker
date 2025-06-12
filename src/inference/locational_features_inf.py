import pandas as pd

def get_locational_features(targets_df):

    # read in locational_variables_latest.csv from the data folder
    loc_df = pd.read_csv("data/locational_variables_latest.csv")

    # make sure sitecode is a string in both targets_df and loc_df
    targets_df["sitecode"] = targets_df["sitecode"].astype(str)
    loc_df["sitecode"] = loc_df["sitecode"].astype(str)
    # now merge the locational features with the targets_df by sitecode
    targets_df = targets_df.merge(loc_df, on="sitecode", how="left")

    return targets_df
