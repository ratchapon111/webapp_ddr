import psycopg2
import logging
import pandas as pd
from io import BytesIO
from datetime import datetime
from sqlalchemy import create_engine

class DDRSQL:
    def __init__(self, db_uri):
        # Use SQLAlchemy engine instead of psycopg2 connection directly
        self.engine = create_engine(db_uri)

    def get_data(self, table_name):
        """Fetch data from a given table."""
        try:
            # Use SQLAlchemy engine with pandas
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql(query, self.engine)  # Using the SQLAlchemy engine here
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def connect(self):
        """Connect to the database (using the psycopg2 connection for cursor-based operations)."""
        try:
            # Example of using psycopg2 for raw queries if needed (use your actual credentials)
            conn = psycopg2.connect(
                dbname="project",
                user="postgres",
                password="102414",
                host="localhost",
                port="5432"
            )
            return conn
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

    def insert_into(self, table_name, df):
        try:
            conn = self.connect()
            if conn is not None:
                cur = conn.cursor()
                columns = ', '.join(df.columns)  # Get column names from DataFrame
                values_placeholder = ', '.join(['%s'] * len(df.columns))  # Prepare placeholders for values
                for index, row in df.iterrows():
                    cur.execute(
                        f'INSERT INTO {table_name} ({columns}) VALUES ({values_placeholder})',
                        tuple(row)
                    )
                conn.commit()
                conn.close()
                logging.info("Data inserted successfully.")
            else:
                logging.error("Failed to connect to the database.")
        except Exception as e:
            logging.error(f"An error occurred while inserting data: {e}")
    def insert_image_path(self, table_name, img_path, label, confidence):
        try:
            conn = self.connect()
            if conn is not None:
                cur = conn.cursor()

                # Insert the image path along with the label and confidence
                cur.execute(f'''
                    INSERT INTO {table_name} (image_path, label, confidence, prediction_time)
                    VALUES (%s, %s, %s, %s)
                ''', (img_path, label, confidence, datetime.datetime.now()))

                conn.commit()
                conn.close()
                logging.info("Image path and metadata inserted successfully.")
            else:
                logging.error("Failed to connect to the database.")
        except Exception as e:
            logging.error(f"An error occurred while inserting the image path: {e}")


    def delete_data(self, table_name, column_name, value):
        """
        Deletes a row from the specified table where the column matches the given value.
        """
        try:
            conn = self.connect()
            if conn is not None:
                cur = conn.cursor()
                query = f"DELETE FROM {table_name} WHERE {column_name} = %s"
                cur.execute(query, (value,))
                conn.commit()
                conn.close()
                logging.info(f"Row deleted successfully from {table_name} where {column_name} = {value}.")
            else:
                logging.error("Failed to connect to the database.")
        except Exception as e:
            logging.error(f"An error occurred while deleting data: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Instantiate the DDRSQL class
        ddrsql = DDRSQL()

    
    except Exception as e:
        logging.error(f"An error occurred while running the script: {e}")
