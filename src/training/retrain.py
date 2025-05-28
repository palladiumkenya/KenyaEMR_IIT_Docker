# Local application imports
from src.common import get_data
from src.common import clean_data
from src.common import visit_features
from src.common import dem_features
from src.common import create_target
from src.common import target_features
import locational_features
import refresh_model

# general imports
import boto3
import io
import pandas as pd
s3 = boto3.client('s3')

# time how long it takes to run the script
import time
start_time = time.time()

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data.get_data(aws = True, prediction = False)

# Run cleaning and feature preparation functions
lab = clean_data.clean_lab(lab, start_date = "2021-01-01")
buffer = io.BytesIO()
lab.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='lab0515.parquet', Body=buffer.getvalue())
print("lab cleaned")
print(time.time() - start_time)

pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = "2021-01-01", end_date = "2025-01-15")
buffer = io.BytesIO()
pharmacy.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='pharmacy0515.parquet', Body=buffer.getvalue())
print("pharmacy cleaned")
print(time.time() - start_time)

print("cleaning visits")
visits = clean_data.clean_visits(visits, dem, start_date = "2021-01-01", end_date = "2025-01-15")
buffer = io.BytesIO()
visits.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='visits0515.parquet', Body=buffer.getvalue())
print("visits cleaned")
print(time.time() - start_time)

visits = visit_features.prep_visit_features(visits)
buffer = io.BytesIO()
visits.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='visits0515.parquet', Body=buffer.getvalue())
print("visits features prepared")
print(time.time() - start_time)

visits = dem_features.prep_demographics(visits)
buffer = io.BytesIO()
visits.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='visits0515.parquet', Body=buffer.getvalue())
print("demographics features prepared")
print(time.time() - start_time)

targets = create_target.create_target(visits, pharmacy, dem)
buffer = io.BytesIO()
targets.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='targets0515.parquet', Body=buffer.getvalue())
print("targets created")
print(time.time() - start_time)

targets = target_features.prep_target_visit_features(targets, visits)
buffer = io.BytesIO()
targets.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='targets0515.parquet', Body=buffer.getvalue())
print("lateness metrics developed")
print(time.time() - start_time)

targets = target_features.prep_target_pharmacy_features(targets, pharmacy)
buffer = io.BytesIO()
targets.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='targets0515.parquet', Body=buffer.getvalue())
print("pharmacy features developed")
print(time.time() - start_time)

targets = target_features.prep_target_lab_features(targets, lab)
buffer = io.BytesIO()
targets.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='targets0515.parquet', Body=buffer.getvalue())
print("lab features developed")
print(time.time() - start_time)

targets = locational_features.prep_locational_features(targets, mfl, dhs, txcurr)
buffer = io.BytesIO()
targets.to_parquet(buffer, index=False)
s3.put_object(Bucket='kehmisjan2025', Key='targets0515.parquet', Body=buffer.getvalue())
print("locational features developed")
print(time.time() - start_time)

# if running in pipeline, then targets_df = targets and pipeline = True.
# if running from AWS, then targets_aws is the filename and pipeline = False.
refresh_model.refresh_model(pipeline = False, targets_df = targets, refresh_date = "2024-09-30")

# end time
end_time = time.time()
print("Time taken to run the script: ", end_time - start_time, " seconds")