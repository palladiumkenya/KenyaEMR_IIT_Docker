import sqlite3
import pandas as pd

def get_data(prediction = False, patientPK = None, sitecode = None):

    # Connect to the SQLite database (or create it if it doesn't exist)
    connection = sqlite3.connect('./data/iit_test.sqlite')

    if prediction:

        # Create a cursor object to interact with the database
        cursor = connection.cursor()

        # Define the SQL query to fetch data from the 'lab' table
        query = "SELECT * FROM lab WHERE PatientPKHash = ? AND SiteCode = ?"
        
        # Execute the query with parameters
        cursor.execute(query, (patientPK, sitecode))

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        lab = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'pharmacy' table
        query = "SELECT * FROM pharmacy WHERE PatientPKHash = ? AND SiteCode = ?"
        
        # Execute the query with parameters
        cursor.execute(query, (patientPK, sitecode))

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        pharmacy = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'visits' table
        query = "SELECT * FROM visits WHERE PatientPKHash = ? AND SiteCode = ?"
        
        # Execute the query with parameters
        cursor.execute(query, (patientPK, sitecode))

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        visits = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'dem' table
        query = "SELECT * FROM dem WHERE PatientPKHash = ? AND MFLCode = ?"
        
        # Execute the query with parameters
        cursor.execute(query, (patientPK, sitecode))

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        dem = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    else:
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
        pharmacy = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'visits' table
        query = "SELECT * FROM visits"
        
        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        visits = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

        # Define the SQL query to fetch data from the 'dem' table
        query = "SELECT * FROM dem"
        
        # Execute the query with parameters
        cursor.execute(query)

        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # Create a DataFrame from the fetched rows
        dem = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    # Pull locational data

    # Pull MFL data
    query = "SELECT * FROM mfl"
    
    # Execute the query with parameters
    cursor.execute(query)

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Create a DataFrame from the fetched rows
    mfl = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])
    
    # Pull MFL data
    query = "SELECT * FROM dhs"
    
    # Execute the query with parameters
    cursor.execute(query)

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Create a DataFrame from the fetched rows
    dhs = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    # Pull MFL data
    query = "SELECT * FROM txcurr"
    
    # Execute the query with parameters
    cursor.execute(query)

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Create a DataFrame from the fetched rows
    txcurr = pd.DataFrame(rows, columns=[column[0] for column in cursor.description])

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return lab, pharmacy, visits, dem, mfl, dhs, txcurr