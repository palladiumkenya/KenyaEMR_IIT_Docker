import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import random
import io
import os
import pandas as pd
import pickle
from src.common.feature_dtypes import expected_dtypes


def gen_inference(df):

    if df is None or df.empty:
        return {"pred_out": None, "pred_cat": "unavailable"}

    required_cols = ["nad", "iit"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing required column(s): {missing} — cannot proceed with inference.")
        return {"pred_out": None, "pred_cat": "unavailable"}

    # make sure nad is a datetime
    df["nad"] = pd.to_datetime(df["nad"], format="%Y-%m-%d")
    # make sure data is sorted by nad in descending order
    df = df.sort_values(by="nad", ascending=False)

    try:
        df = df.drop(
            columns=[
                "key", "visitdate", "nad_imputation_flag", "sitecode",
                "pregnant_missing", "nad", "breastfeeding_missing",
                "startartdate", "month", "dayofweek", "timeatfacility",
            ],
            errors="ignore"  # safer option than try/except if you’re okay silently skipping
        )
    except KeyError as e:
        print(f"⚠️ Unexpected missing columns during drop: {e}")

    # filter to emr in kenyamer and ecare
    df = df[df["emr"].isin(["kenyaemr", "ecare"])]
    # Emr: KenyaEMR -> 1, else 0
    df["emr"] = (df["emr"] == "kenyaemr").astype("Int64")

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ensure columns are right dtypes
    # for col, dtype in expected_dtypes.items():
    #     if col in df.columns:
    #         df[col] = df[col].astype(dtype)

    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            if dtype in [float, "float", "float64", int, "int", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
            # try:
            #     print(f"Converting column '{col}' to {dtype}...")
            #     df[col] = df[col].astype(dtype)
            # except Exception as e:
            #     print(f"❌ Error converting column '{col}' to {dtype}")
            #     print("Unique values in the column:", df[col].unique()[:10])  # show a few
            #     raise e  # re-raise the error to preserve traceback

    # load encoder which is called ohe_latest.pkl
    # from the models directory
    encoder = "models/ohe_latest.pkl"
    # Check if the encoder file exists
    if not os.path.exists(encoder):
        raise FileNotFoundError(
            f"Encoder file {encoder} not found. Please train the model first."
        )
    with open(encoder, "rb") as f:
        ohe = pickle.load(f)

    # encode categorical columns
    # Get the categorical columns from the DataFrame
    # Note: This assumes that the categorical columns are the same as those used during training
    # If the columns are different, you may need to adjust this part
    # to match the training columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # One-hot encode the categorical columns
    try:
        encoded_features = ohe.transform(df[categorical_columns]).toarray()
    except Exception as e:
        print(f"OneHotEncoding failed: {e}")
        return {"pred_out": None, "pred_cat": "unavailable"}
    encoded_feature_names = ohe.get_feature_names_out(categorical_columns)

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(
        encoded_features, columns=encoded_feature_names, index=df.index
    )

    # Concatenate the encoded features with the original DataFrame
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

    # make sure the columns are in the right order
    with open("models/feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
    try:
        final_df = final_df[feature_order]
    except KeyError as e:
        print(f"❌ Feature mismatch: some expected columns are missing: {e}")
        return {"pred_out": None, "pred_cat": "unavailable"}

    # convert to xgb.Dmatrix
    xgb_df = xgb.DMatrix(data=final_df.drop(columns=["iit"]), label=final_df["iit"])

    # load model
    model = "models/mod_latest.json"
    # Check if the model file exists
    if not os.path.exists(model):
        raise FileNotFoundError(
            f"Model file {model} not found. Please train the model first."
        )
    bst = xgb.Booster()
    bst.load_model(model)

    # make prediction
    try:
        preds = bst.predict(xgb_df)
        pred_out = preds[0].item()
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return {"pred_out": None, "pred_cat": "unavailable"}

    # load thresholds from models/thresholds.pkl
    thresholds_file = "models/thresholds_latest.pkl"
    if not os.path.exists(thresholds_file):
        raise FileNotFoundError(
            f"Thresholds file {thresholds_file} not found. Please train the model first."
        )
    with open(thresholds_file, "rb") as f:
        thresholds = pickle.load(f)

    # apply thresholds to pred_cat
    # if pred is greater than thresholds['high'], pred_cat returns 'high',
    # else if pred is greater than thresholds['medium'], return 'medium',
    # else return 'low'
    if pred_out > thresholds["high"]:
        pred_cat = "high"
    elif pred_out > thresholds["medium"]:
        pred_cat = "medium"
    else:
        pred_cat = "low"

    # if pred_cat is high or medium, return risk factors from final_df including:
    # if lateness_last5 is greater than 0, return lateness_last5,
    # if most_recent_vl is "unsuppressed", return "unsuppressed",
    if pred_cat in ["high", "medium"]:
        adherence_val = final_df["adherence"].iloc[0]
        if pd.isna(adherence_val):
            adherence = None
        elif adherence_val == 1:
            adherence = "good"
        elif adherence_val == 0:
            adherence = "poor"
        else:
            adherence = None
        risk_factors = {
            "avg_days_late_last5visits": final_df["lateness_last5"].iloc[0],
            "months_on_art": final_df["timeonart"].iloc[0],
            "most_recent_viralload": df["most_recent_vl"].iloc[0],
            # if adherence is 1, then return "good", if 0, return "poor", otherwise None
            # "adherence": "good" if final_df["adherence"].iloc[0] == 1 else "poor" if final_df["adherence"].iloc[0] == 0 else None,
            "adherence": adherence,
            # if visittype is 1, then return "unscheduled visits", otherwise return "no unscheduled visits"
            "unscheduled_visits": "unscheduled visits" if final_df["visittype"].iloc[0] == 1 else "no unscheduled visits",
        }
    else:
        risk_factors = None

    # return pred_out and pred_cat
    pred_out = {
        "pred_out": pred_out,
        "pred_cat": pred_cat,
        "risk_factors": risk_factors,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
    }
    return pred_out
