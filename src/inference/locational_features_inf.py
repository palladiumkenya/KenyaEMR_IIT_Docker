import boto3
import io
import pandas as pd


def get_locational_features(targets_df):

    # connect to s3
    s3 = boto3.client("s3")
    bucket = "kehmisjan2025"
    key = "locational_variables_latest.csv"

    # download the file into memory
    obj = s3.get_object(Bucket=bucket, Key=key)
    loc_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    # make sure sitecode is a string in both targets_df and loc_df
    targets_df["sitecode"] = targets_df["sitecode"].astype(str)
    loc_df["sitecode"] = loc_df["sitecode"].astype(str)
    # now merge the locational features with the targets_df by sitecode
    targets_df = targets_df.merge(loc_df, on="sitecode", how="left")

    return targets_df
