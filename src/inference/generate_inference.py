import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import random
import boto3
import io
import os
import pandas as pd
import pickle
from src.common.feature_dtypes import expected_dtypes

def gen_inference(df):

    # make sure nad is a datetime
    df['nad'] = pd.to_datetime(df['nad'], format='%Y-%m-%d')
    # make sure data is sorted by nad in descending order
    df = df.sort_values(by='nad', ascending=False)

    df = df.drop(columns=[
        'key', 'visitdate', 'nad_imputation_flag', 'sitecode', 'pregnant_missing', 'nad',
        'breastfeeding_missing', 'startartdate', 'month', 'dayofweek', 'timeatfacility'])

    # filter to emr in kenyamer and ecare   
    df = df[df['emr'].isin(['kenyaemr', 'ecare'])]
    # Emr: KenyaEMR -> 1, else 0
    df['emr'] = (df['emr'] == 'kenyaemr').astype('Int64') 

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # ensure columns are right dtypes
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # load encoder which is called ohe_latest.pkl
    # from the models directory
    encoder = "models/ohe_latest.pkl"
    # Check if the encoder file exists
    if not os.path.exists(encoder):
        raise FileNotFoundError(f"Encoder file {encoder} not found. Please train the model first.")
    with open(encoder, "rb") as f:
        ohe = pickle.load(f)

    # encode categorical columns
    # Get the categorical columns from the DataFrame
    # Note: This assumes that the categorical columns are the same as those used during training
    # If the columns are different, you may need to adjust this part
    # to match the training columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode the categorical columns
    encoded_features = ohe.transform(df[categorical_columns]).toarray()
    encoded_feature_names = ohe.get_feature_names_out(categorical_columns)
    
    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
    # Concatenate the encoded features with the original DataFrame
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

    # make sure the columns are in the right order
    with open("models/feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
    final_df = final_df[feature_order]

    # convert to xgb.Dmatrix
    xgb_df = xgb.DMatrix(
        data=final_df.drop(columns=["iit"]),
        label=final_df["iit"]
    )

    # load model
    model = "models/mod_latest.json"
    # Check if the model file exists
    if not os.path.exists(model):
        raise FileNotFoundError(f"Model file {model} not found. Please train the model first.")
    bst = xgb.Booster()
    bst.load_model(model)

    # make prediction
    preds = bst.predict(xgb_df)

    # return prediction that is first item in preds
    return preds[0]
