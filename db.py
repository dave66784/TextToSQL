from typing import Any, List, Tuple

import psycopg2
import pandas as pd

from config import DB_STANDARD


class SQLDatabase:
    def __init__(self, conn_params: dict | None = None) -> None:
        self.conn_params = conn_params or DB_STANDARD

    def _connect(self):
        return psycopg2.connect(
            host=self.conn_params["host"],
            port=self.conn_params["port"],
            dbname=self.conn_params["dbname"],
            user=self.conn_params["user"],
            password=self.conn_params["password"],
        )

    def run_query(self, sql: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn)
        return df


