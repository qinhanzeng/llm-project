from pgvector.psycopg import register_vector
import psycopg

def get_conn():
    conn = psycopg.connect(
        dbname="postgres",
        host="localhost",
        user="hanson",
        password="",
        port=5432
    )

    register_vector(conn)

    return conn