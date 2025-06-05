import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import random
import boto3
import io
import pandas as pd
import pickle
import shutil
from src.common.feature_dtypes import expected_dtypes


def refresh_model(pipeline=False, targets_df=None, targets_aws=None, refresh_date=str):

    # first, read in the processed dataset
    # if pipeline, then dataset is in the pipeline
    # else, it is in the AWS S3 bucket
    if pipeline:
        df = targets_df
    else:
        # Define S3 info
        bucket = "kehmisjan2025"
        # Initialize boto3 client
        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, targets_aws, buffer)
        buffer.seek(0)
        df = pd.read_parquet(buffer)

    # make sure nad is a datetime
    df["nad"] = pd.to_datetime(df["nad"], format="%Y-%m-%d")

    # filter to refresh period
    refresh_date = pd.Timestamp(refresh_date)

    # Filter to records from the refresh date and six months before
    # Define the date range to exclude
    after = refresh_date - pd.DateOffset(months=6)
    before = refresh_date
    df = df[(df["nad"] >= after) & (df["nad"] <= before)]

    # filter out where nad imputation flag is 1
    df = df[df["nad_imputation_flag"] == 0]

    df = df.drop(
        columns=[
            "key",
            "visitdate",
            "nad_imputation_flag",
            "sitecode",
            "pregnant_missing",
            "breastfeeding_missing",
            "startartdate",
            "month",
            "dayofweek",
            "timeatfacility",
            "code",
            "county",
        ]
    )

    # filter to emr in kenyamer and ecare
    df = df[df["emr"].isin(["kenyaemr", "ecare"])]
    # Emr: KenyaEMR -> 1, else 0
    df["emr"] = (df["emr"] == "kenyaemr").astype("Int64")

    # make sure all column names are lowercase and no whitespace
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ensure columns are right dtypes
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    ohe.fit(df[categorical_columns])

    # Save the fitted encoder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"models/ohe_{timestamp}.pkl", "wb") as f:
        pickle.dump(ohe, f)
    # Save the refreshed encoder as latest to be used in inference
    shutil.copyfile(f"models/ohe_{timestamp}.pkl", "models/ohe_latest.pkl")

    def encode_xgboost(df, start_date, end_date):

        # Filter the DataFrame to include only the rows within the specified date range
        df = df[(df["nad"] >= start_date) & (df["nad"] <= end_date)]
        # Drop the 'nad' column from the DataFrame
        df = df.drop(columns=["nad"])

        # One-hot encode the categorical columns
        encoded_features = ohe.transform(df[categorical_columns]).toarray()
        encoded_feature_names = ohe.get_feature_names_out(categorical_columns)

        # Create a DataFrame with the encoded features
        encoded_df = pd.DataFrame(
            encoded_features, columns=encoded_feature_names, index=df.index
        )

        # Concatenate the encoded features with the original DataFrame
        final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

        feature_order = list(final_df.columns)
        with open("models/feature_order.pkl", "wb") as f:
            pickle.dump(feature_order, f)

        # convert to xgb.Dmatrix
        xgb_df = xgb.DMatrix(data=final_df.drop(columns=["iit"]), label=final_df["iit"])

        return xgb_df

    # encoded dataset
    dtrain = encode_xgboost(
        df, start_date=after, end_date=refresh_date - pd.DateOffset(months=1)
    )
    dval = encode_xgboost(
        df, start_date=refresh_date - pd.DateOffset(months=1), end_date=refresh_date
    )

    params = {
        "eta": 0.01,
        "max_depth": 6,
        "subsample": 0.5,
        "colsample_bytree": 0.6,
        "lambda": 1,
        "scale_pos_weight": 10,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
    }

    random.seed(42)
    gb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # After training with xgb.train(...)
    gb_model.save_model(f"models/mod_{timestamp}.json")
    shutil.copyfile(f"models/mod_{timestamp}.json", "models/mod_latest.json")


if __name__ == "__main__":
    refresh_model(
        pipeline=False, targets_aws="targets0521.parquet", refresh_date="2024-09-30"
    )
