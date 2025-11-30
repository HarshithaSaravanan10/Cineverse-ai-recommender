import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Harsha@45",  # Change
        database="movie_recommender"
    )
