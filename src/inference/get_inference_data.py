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

def mysql_connect():
    config = load_settings()
    connection = None

    try:
        connection = mysql.connector.connect(
            host=config["mysql_url"],
            port=int(config["mysql_port"]),
            database=config["mysql_database"],
            user=config["mysql_username"],
            password=config["mysql_password"]
        )
    except Error as e:
        print(f"MySQL Error: {e}")

    return connection

def get_inference_data_mysql(patientPK=None, sitecode=None):
    # Initialize variables to None
    pharmacy = lab = visits = dem = None
    # # load settings
    # config = load_settings()

    try:
        # connection = mysql.connector.connect(
        #     host=config["mysql_url"],
        #     port=int(config["mysql_port"]),
        #     database=config["mysql_database"],
        #     user=config["mysql_username"],
        #     password=config["mysql_password"]
        # )

        connection = mysql_connect()

        if connection.is_connected():
            print("Connected to MySQL For Inference 1 - Lab")

            labCursor = connection.cursor(dictionary=True)

            # Lab Table
            print("Lab Table: ")
            labQuery = "CALL sp_iitml_get_patient_lab(%s)"
            labCursor.execute(labQuery, (patientPK, ))
            # Fetch all rows from the executed query
            labRows = labCursor.fetchall()
            # Create a DataFrame from the fetched rows
            # if rows is empty, return empty DataFrame, set lab to an empty dataframe
            if not labRows:
                lab = pd.DataFrame(columns=[column[0] for column in labCursor.description])
            else:
                # Create a DataFrame from the fetched rows
                lab = pd.DataFrame(labRows, columns=[column[0] for column in labCursor.description])
            # close cursor
            labCursor.close()

    except Error as e:
        print(f"MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            labCursor.close()
            connection.close()
            print("Mysql Connection closed.")

    try:
        # connection = mysql.connector.connect(
        #     host=config["mysql_url"],
        #     port=int(config["mysql_port"]),
        #     database=config["mysql_database"],
        #     user=config["mysql_username"],
        #     password=config["mysql_password"]
        # )

        connection = mysql_connect()

        if connection.is_connected():
            print("Connected to MySQL For Inference 1 - Pharmacy")

            pharmacyCursor = connection.cursor(dictionary=True)

            # Pharmacy Table
            print("Pharmacy Table: ")
            pharmacyQuery = "CALL sp_iitml_get_pharmacy_visits(%s)"
            pharmacyCursor.execute(pharmacyQuery, (patientPK, ))
            # Fetch all rows from the executed query
            pharmacyRows = pharmacyCursor.fetchall()
            # Create a DataFrame from the fetched rows
            if not pharmacyRows:
                pharmacy = pd.DataFrame(columns=[column[0] for column in pharmacyCursor.description])
            else:
                # Create a DataFrame from the fetched rows
                pharmacy = pd.DataFrame(pharmacyRows, columns=[column[0] for column in pharmacyCursor.description])

    except Error as e:
        print(f"MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            pharmacyCursor.close()
            connection.close()
            print("Mysql Connection closed.")

    try:
        # connection = mysql.connector.connect(
        #     host=config["mysql_url"],
        #     port=int(config["mysql_port"]),
        #     database=config["mysql_database"],
        #     user=config["mysql_username"],
        #     password=config["mysql_password"]
        # )

        connection = mysql_connect()

        if connection.is_connected():
            print("Connected to MySQL For Inference 1 - Visits")

            visitCursor = connection.cursor(dictionary=True)

            # Visits Table
            print("Visits Table: ")
            # visitQueryBuilder = ['CALL sp_iitml_get_visits("', patientPK, '");']
            # visitQuery = "".join(visitQueryBuilder)
            # print("Mysql using the query: " + visitQuery)
            # cursor.execute(visitQuery)
            visitQuery = "CALL sp_iitml_get_visits(%s)"
            visitCursor.execute(visitQuery, (patientPK, ))
            # Fetch all rows from the executed query
            visitRows = visitCursor.fetchall()
            # Create a DataFrame from the fetched rows
            if not visitRows:
                visits = pd.DataFrame(columns=[column[0] for column in visitCursor.description])
            else:
                # Create a DataFrame from the fetched rows
                visits = pd.DataFrame(
                    visitRows, columns=[column[0] for column in visitCursor.description]
                )

    except Error as e:
        print(f"MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            visitCursor.close()
            connection.close()
            print("Mysql Connection closed.")

    try:
        # connection = mysql.connector.connect(
        #     host=config["mysql_url"],
        #     port=int(config["mysql_port"]),
        #     database=config["mysql_database"],
        #     user=config["mysql_username"],
        #     password=config["mysql_password"]
        # )

        connection = mysql_connect()

        if connection.is_connected():
            print("Connected to MySQL For Inference 1 - Demographics")

            demCursor = connection.cursor(dictionary=True)

            # Dem (Demographics) Table
            print("Demographics Table: ")
            demQuery = "CALL sp_iitml_get_patient_demographics(%s)"
            demCursor.execute(demQuery, (patientPK, ))
            # Fetch all rows from the executed query
            demRows = demCursor.fetchall()
            # Create a DataFrame from the fetched rows
            if not demRows:
                dem = pd.DataFrame(columns=[column[0] for column in demCursor.description])
            else:
                # Create a DataFrame from the fetched rows
                dem = pd.DataFrame(demRows, columns=[column[0] for column in demCursor.description])

    except Error as e:
        print(f"MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            demCursor.close()
            connection.close()
            print("Mysql Connection closed.")

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
