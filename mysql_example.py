import json
import mysql.connector
from mysql.connector import Error
import pandas as pd

def load_settings(path='data/settings.json'):
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

def connect_and_query():
    # config = load_settings()
    patientPK = 67

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
            print("Connected to MySQL 8")

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username FROM users LIMIT 5;")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

            # Lab Table
            print("Lab Table: ")
            labCursor = connection.cursor(dictionary=True)
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

    except Error as e:
        print(f"MySQL Error: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed.")

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
            print("Connected to MySQL 8")

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT user_id, username FROM users LIMIT 5;")
            rows = cursor.fetchall()

            for row in rows:
                print(row)

            # Pharmacy Table
            print("Pharmacy Table: ")
            
            pharmacyCursor = connection.cursor(dictionary=True)
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
            cursor.close()
            connection.close()
            print("Connection closed.")

if __name__ == '__main__':
    connect_and_query()
