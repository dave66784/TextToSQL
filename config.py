# config.py

DB_STANDARD = {
    "host": "localhost",
    "port": 5432,  # Standard PostgreSQL instance
    "dbname": "mydatabase",
    "user": "myuser",
    "password": "mypassword"
}

DB_VECTOR = {
    "host": "localhost",
    "port": 5433,  # pgvector-enabled PostgreSQL instance
    "dbname": "vectordb",
    "user": "vectoruser",
    "password": "vectorpass"
}