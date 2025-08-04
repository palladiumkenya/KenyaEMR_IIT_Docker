import sqlite3
import pandas as pd
import boto3
import pyreadr
import tempfile
import json
import mysql.connector
from mysql.connector import Error

def load_settings(path='/data/settings.json'):
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load settings: {e}")

def get_training_data_mysql(aws=False):
    config = load_settings()

    try:
        connection = mysql.connector.connect(
            host=config["mysql_url"],
            port=int(config["mysql_port"]),
            database=config["mysql_database"],
            user=config["mysql_username"],
            password=config["mysql_password"]
        )

        if connection.is_connected():
            print("Connected to MySQL For Training")

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username FROM users LIMIT 5;")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

    except Error as e:
        print(f"MySQL Error: {e}")

    return lab, pharmacy, visits, dem, mfl, dhs, txcurr

def get_inference_data_mysql(patientPK=None, sitecode=None):
    config = load_settings()

    try:
        connection = mysql.connector.connect(
            host=config["mysql_url"],
            port=int(config["mysql_port"]),
            database=config["mysql_database"],
            user=config["mysql_username"],
            password=config["mysql_password"]
        )

        if connection.is_connected():
            print("Connected to MySQL For Inference 2")

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username FROM users LIMIT 5;")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

    except Error as e:
        print(f"MySQL Error: {e}")

    return lab, pharmacy, visits, dem

def get_training_data_sqlite(aws=False):

    # Initialize variables to None
    pharmacy = lab = visits = dem = mfl = dhs = txcurr = None

    # If aws is True, then read in pharmacy, lab, visits and dem data from S3
    if aws:

        s3 = boto3.client("s3")
        bucket_name = "kehmisjan2025"

        # Define the list of files to download
        files = [
            "pharmacy_all_feb2025.rds",
            "labs_all_feb2025.rds",
            "visits_all_feb2025.rds",
            "dem_all_may2025.rds",
        ]

        # Now, read them in one by one and store them in separate dataframes called pharmacy, lab, visits, dem
        for file_key in files:
            with tempfile.NamedTemporaryFile(suffix=".rds") as tmp_file:
                print(file_key)
                s3.download_fileobj(Bucket=bucket_name, Key=file_key, Fileobj=tmp_file)
                tmp_file.seek(0)
                result = pyreadr.read_r(tmp_file.name)

                # Extract the DataFrame from the result
                if file_key == "pharmacy_all_feb2025.rds":
                    pharmacy = result[None]
                elif file_key == "labs_all_feb2025.rds":
                    lab = result[None]
                elif file_key == "visits_all_feb2025.rds":
                    visits = result[None]
                elif file_key == "dem_all_may2025.rds":
                    dem = result[None]

        # Check that all variables are loaded
        if any(x is None for x in [lab, pharmacy, visits, dem]):
            raise ValueError(
                "One or more dataframes (lab, pharmacy, visits, dem) were not loaded from S3. Check file names and S3 bucket."
            )

    # if aws if False, read in pharmacy, lab, visits and dem data from SQLite
    else:

        # Create a connection to the SQLite database (or create it if it doesn't exist)
        connection = sqlite3.connect("./data/iit_test.sqlite")

        # Create a cursor object to interact with the database
        cursor = connection.cursor()

        # Define the SQL query to fetch data from the 'lab' table
        query = "SELECT * FROM lab"

        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        lab = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'pharmacy' table
        query = "SELECT * FROM pharmacy"

        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        pharmacy = pd.DataFrame(
            rows, columns=[column[0] for column in cursor.description]
        )

        # Define the SQL query to fetch data from the 'visits' table
        query = "SELECT * FROM visits"

        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        visits = pd.DataFrame(
            rows, columns=[column[0] for column in cursor.description]
        )

        # Define the SQL query to fetch data from the 'dem' table
        query = "SELECT * FROM dem"

        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        dem = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    # Pull locational data
    # Create a connection to the SQLite database (or create it if it doesn't exist)
    connection = sqlite3.connect("./data/iit_test.sqlite")
    cursor = connection.cursor()
    for table in ["mfl", "dhs", "txcurr"]:
        query = f"SELECT * FROM {table}"
        cursor.execute(query)
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])
        if table == "mfl":
            mfl = df
        elif table == "dhs":
            dhs = df
        elif table == "txcurr":
            txcurr = df
    cursor.close()
    connection.close()

    return lab, pharmacy, visits, dem, mfl, dhs, txcurr


def get_inference_data_sqlite(patientPK=None, sitecode=None):

    # Initialize variables to None
    pharmacy = lab = visits = dem = None
    # Create a connection to the SQLite database (or create it if it doesn't exist)
    connection = sqlite3.connect("./data/iit_test.sqlite")
    # Create a cursor object to interact with the database
    cursor = connection.cursor()
    # Define the SQL query to fetch data from the 'lab' table
    query = "SELECT * FROM lab WHERE PatientPKHash = ? AND SiteCode = ?"
    # Execute the query with parameters
    cursor.execute(query, (patientPK, sitecode))
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    # Create a DataFrame from the fetched rows
    # if rows is empty, return empty DataFrame, set lab to an empty dataframe
    if not rows:
        lab = pd.DataFrame(columns=[column[0] for column in cursor.description])
    else:
        # Create a DataFrame from the fetched rows
        lab = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    # Define the SQL query to fetch data from the 'pharmacy' table
    query = "SELECT * FROM pharmacy WHERE PatientPKHash = ? AND SiteCode = ?"
    # Execute the query with parameters
    cursor.execute(query, (patientPK, sitecode))
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    # Create a DataFrame from the fetched rows
    if not rows:
        pharmacy = pd.DataFrame(columns=[column[0] for column in cursor.description])
    else:
        # Create a DataFrame from the fetched rows
        pharmacy = pd.DataFrame(
            rows, columns=[column[0] for column in cursor.description]
        )

    # Define the SQL query to fetch data from the 'visits' table
    query = "SELECT * FROM visits WHERE PatientPKHash = ? AND SiteCode = ?"
    # Execute the query with parameters
    cursor.execute(query, (patientPK, sitecode))
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    # Create a DataFrame from the fetched rows
    if not rows:
        visits = pd.DataFrame(columns=[column[0] for column in cursor.description])
    else:
        # Create a DataFrame from the fetched rows
        visits = pd.DataFrame(
            rows, columns=[column[0] for column in cursor.description]
        )

    # Define the SQL query to fetch data from the 'dem' table
    # Execute the query with parameters
    query = "SELECT * FROM dem WHERE PatientPKHash = ? AND MFLCode = ?"
    cursor.execute(query, (patientPK, sitecode))
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    # Create a DataFrame from the fetched rows
    if not rows:
        dem = pd.DataFrame(columns=[column[0] for column in cursor.description])
    else:
        # Create a DataFrame from the fetched rows
        dem = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    return lab, pharmacy, visits, dem
