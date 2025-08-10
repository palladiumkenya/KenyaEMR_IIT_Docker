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

    # filter to emr in kenyamer and ecare
    df = df[df["emr"].isin(["kenyaemr", "ecare"])]
    # Emr: KenyaEMR -> 1, else 0
    df["emr"] = (df["emr"] == "kenyaemr").astype("Int64")

    # get each patientpkhash and sitecode and save to file
    df = df.drop(
        columns=[
            "key",
            "visitdate",
            "nad_imputation_flag",
            # "sitecode",
            "pregnant_missing",
            "breastfeeding_missing",
            "startartdate",
            "month",
            "dayofweek",
            "timeatfacility",
            "code",
            "county",
            "txcurr",
            "rolling_weighted_noshow",
            "rolling_weighted_dayslate",
            # "kephlevel",
            # "facilitytypecategory",
            # "ownertype",
            # "men_knowledge",
            # "women_knowledge",
            # "men_heardaids",
            # "men_highrisksex",
            # "men_highrisksex_multi",
            # "men_sexnotwithpartner",
            # "men_sexpartners",
            # "men_nevertested",
            # "men_testedrecent",
            # "men_sti",
            # "women_highrisksex",
            # "women_highrisksex_multi",
            # "women_sexnotwithpartner",
            # "women_sexpartners",
            # "women_nevertested",
            # "women_testedrecent",
            # "women_sti"
        ]
    )

    # make sure all column names are lowercase and no whitespace
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ensure columns are right dtypes
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_columns = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in ("sitecode", "iit")
    ]

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    ohe.fit(df[categorical_columns])

    # Save the fitted encoder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"models/ohe_{timestamp}.pkl", "wb") as f:
        pickle.dump(ohe, f)
    # Save the refreshed encoder as latest to be used in inference
    shutil.copyfile(f"models/ohe_{timestamp}.pkl", "models/ohe_latest.pkl")

    def encode_xgboost(df, start_date, end_date, save_feature_order):

        # Filter the DataFrame to include only the rows within the specified date range
        # slice by date
        mask = (df["nad"] >= start_date) & (df["nad"] <= end_date)
        df_slice = df.loc[mask].copy()

        # stash sitecodes in the same row order as features/preds
        sitecodes = df_slice["sitecode"].values

        # drop non-feature cols before encoding
        df_slice = df_slice.drop(columns=["nad", "sitecode"])

        # one-hot encode categorical cols (may be empty)
        if categorical_columns:
            encoded = ohe.transform(df_slice[categorical_columns]).toarray()
            encoded_cols = ohe.get_feature_names_out(categorical_columns)
            enc_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_slice.index)
            final_df = pd.concat([df_slice.drop(columns=categorical_columns), enc_df], axis=1)
        else:
            final_df = df_slice

        feature_order = list(final_df.columns)
        if save_feature_order:
            with open("models/feature_order.pkl", "wb") as f:
                pickle.dump(feature_order, f)

        # convert to xgb.Dmatrix
        xgb_df = xgb.DMatrix(data=final_df.drop(columns=["iit"]), label=final_df["iit"])

        return xgb_df, sitecodes

    # encoded dataset
    dtrain, _ = encode_xgboost(
        df, start_date=after, end_date=refresh_date - pd.DateOffset(months=1), save_feature_order=True
    )
    dval, dval_sitecodes = encode_xgboost(
        df, start_date=refresh_date - pd.DateOffset(months=1), end_date=refresh_date, save_feature_order=False
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

    # Generate predictions on the validation set
    preds = gb_model.predict(dval)
    # save preds to a csv file
    preds_df = pd.DataFrame({"sitecode": dval_sitecodes, "preds": preds})

    # Let's set global thresholds for low-volume sites and site-specific thresholds
    # for the sites with the 100 highest volume of patients
    # get the 100 highest volume sites
    top_sites = df["sitecode"].value_counts().nlargest(100).index.tolist()
    top_preds = preds_df[preds_df["sitecode"].isin(top_sites)]

    # for each of these sites, get the 75th and 50th percentiles of the predictions
    site_thresholds = {}
    for site in top_sites:
        site_preds = top_preds[top_preds["sitecode"] == site]["preds"]
        if not site_preds.empty:
            threshold_high = site_preds.quantile(0.75)
            threshold_medium = site_preds.quantile(0.5)
            site_thresholds[site] = {
                "high": threshold_high,
                "medium": threshold_medium,
            }
    
    # now, get global thresholds for low-volume sites and add them to site_thresholds
    global_thresholds = {
        "high": pd.Series(preds).quantile(0.75),
        "medium": pd.Series(preds).quantile(0.5),
    }
    
    # add global thresholds to site_thresholds for low-volume sites
    for site in df["sitecode"].unique():
        if site not in site_thresholds:
            site_thresholds[site] = global_thresholds         

    # for each row in preds_df, assign the site-specific thresholds
    preds_df["thresholds"] = preds_df["sitecode"].map(site_thresholds)

    # apply thresholds to pred_cat
    def categorize_prediction(row):
        pred = row["preds"]
        thresholds = row["thresholds"]
        if pred > thresholds["high"]:
            return "high"
        elif pred > thresholds["medium"]:
            return "medium"
        else:
            return "low"

    preds_df["pred_cat"] = preds_df.apply(categorize_prediction, axis=1)
    # save preds_df to a csv file with timestamp and as latest
    preds_df.to_csv(f"models/preds_{timestamp}.csv", index=False)

    # save the site-specific thresholds to a file with timestamp and as latest
    with open(f"models/site_thresholds_{timestamp}.pkl", "wb") as f:
        pickle.dump(site_thresholds, f)
    shutil.copyfile(f"models/site_thresholds_{timestamp}.pkl", "models/site_thresholds_latest.pkl")


    # get the 25th percentile of the predictions
    threshold_high = pd.Series(preds).quantile(0.75)
    threshold_medium = pd.Series(preds).quantile(0.5)
    print(f"Thresholds: high={threshold_high}, medium={threshold_medium}")
    # combine thresholds into a dictionary
    thresholds = {
        "high": threshold_high,
        "medium": threshold_medium,
    }   
    # save thresholds to a file with timestamp and as latest
    with open(f"models/thresholds_{timestamp}.pkl", "wb") as f:
        pickle.dump(thresholds, f)
    shutil.copyfile(f"models/thresholds_{timestamp}.pkl", "models/thresholds_latest.pkl")


if __name__ == "__main__":
    refresh_model(
        pipeline=False, targets_aws="targets0521.parquet", refresh_date="2024-09-30"
    )
