# let's test reading in a file from s3
import boto3
import pandas as pd
import pyreadr
import tempfile

s3 = boto3.client("s3")

# Example: list buckets
buckets = s3.list_buckets()
for bucket in buckets["Buckets"]:
    print(bucket["Name"])

bucket_name = 'kehmisjan2025'
file_key = 'visits_clean_apr2.rds'

response = s3.head_object(Bucket='kehmisjan2025', Key='visits_clean_apr2.rds')
print(f"File size: {response['ContentLength'] / (1024**2):.2f} MB")


with tempfile.NamedTemporaryFile(suffix=".rds") as tmp_file:
    s3.download_fileobj(Bucket=bucket_name, Key=file_key, Fileobj=tmp_file)
    tmp_file.seek(0)
    result = pyreadr.read_r(tmp_file.name)

def read_s3_file(bucket_name, file_key):
    """
    Read an RDS file from S3 and return it as a pandas DataFrame.
    """
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')


    # Download the RDS file from S3 to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.rds') as tmp_file:
        s3.download_fileobj(Bucket=bucket_name, Key=file_key, Fileobj=tmp_file)
        tmp_file.seek(0)  # go back to the beginning of the file

        # Read RDS using pyreadr
        result = pyreadr.read_r(tmp_file.name)
    
    return result

# Read the file
df = read_s3_file(bucket_name, file_key)
# Convert the result to a DataFrame
