from __future__ import annotations

from typing import Any, Dict, List
import psycopg2

from config import DB_STANDARD


def fetch_schema_from_postgres(conn_params: dict | None = None, schema: str = "public") -> List[Dict[str, Any]]:
    """Return schema description in the same structure as DB_Schema.json.

    Includes table name, schema, and columns with data_type, nullable, default, and descriptions if available.
    """
    params = conn_params or DB_STANDARD
    with psycopg2.connect(
        host=params["host"],
        port=params["port"],
        dbname=params["dbname"],
        user=params["user"],
        password=params["password"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    pd.description AS column_description
                FROM information_schema.columns c
                JOIN information_schema.tables t
                    ON c.table_schema = t.table_schema AND c.table_name = t.table_name
                LEFT JOIN pg_catalog.pg_class pc
                    ON pc.relname = c.table_name
                LEFT JOIN pg_catalog.pg_namespace pn
                    ON pn.nspname = c.table_schema AND pn.oid = pc.relnamespace
                LEFT JOIN pg_catalog.pg_attribute pa
                    ON pa.attrelid = pc.oid AND pa.attname = c.column_name
                LEFT JOIN pg_catalog.pg_description pd
                    ON pd.objoid = pc.oid AND pd.objsubid = pa.attnum
                WHERE t.table_type = 'BASE TABLE' AND c.table_schema = %s
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
                """,
                (schema,),
            )
            rows = cur.fetchall()

    # Group by table
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for (table_schema, table_name, column_name, data_type, is_nullable, column_default, description) in rows:
        key = (table_schema, table_name)
        grouped.setdefault(key, [])
        grouped[key].append(
            {
                "column_name": column_name,
                "data_type": data_type,
                "is_nullable": is_nullable,
                "column_default": column_default,
                "description": description,
            }
        )

    result: List[Dict[str, Any]] = []
    for (table_schema, table_name), columns in grouped.items():
        result.append({
            "table_schema": table_schema,
            "table_name": table_name,
            "columns": columns,
        })
    return result


