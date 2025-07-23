import sqlite3
import pandas as pd
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
            print("Connected to MySQL 8")

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username FROM users LIMIT 5;")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

    except Error as e:
        print(f"MySQL Error: {e}")

    return lab, pharmacy, visits, dem

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
