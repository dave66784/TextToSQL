import json
from typing import List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

from config import DB_VECTOR


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, conn_params: dict | None = None, embedding_model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.conn_params = conn_params or DB_VECTOR
        self.model = SentenceTransformer(embedding_model_name)

    def _connect(self):
        return psycopg2.connect(
            host=self.conn_params["host"],
            port=self.conn_params["port"],
            dbname=self.conn_params["dbname"],
            user=self.conn_params["user"],
            password=self.conn_params["password"],
        )

    def setup(self) -> None:
        """Create extension, collection table and indexes if not exists."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_chunks (
                  id BIGSERIAL PRIMARY KEY,
                  source TEXT NOT NULL,               -- file/table origin
                  chunk TEXT NOT NULL,                -- natural language chunk
                  metadata JSONB NOT NULL DEFAULT '{}',
                  embedding VECTOR(384)              -- dimension must match model
                );
                """
            )
            # pgvector requires specifying the operator class, e.g., vector_cosine_ops
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_schema_chunks_embedding ON schema_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
                )
            except Exception:
                # Fallback: create a plain index-less table (search still works, just slower)
                pass
            cur.execute("CREATE INDEX IF NOT EXISTS idx_schema_chunks_source ON schema_chunks (source);")
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE schema_chunks;")
            conn.commit()

    def upsert_chunks(self, rows: List[Tuple[str, str, dict]]) -> int:
        """Insert chunks with embeddings. rows: [(source, chunk, metadata_json), ...]"""
        if not rows:
            return 0
        texts = [r[1] for r in rows]
        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
        values = [(r[0], r[1], json.dumps(r[2]), self._to_pgvector(e)) for r, e in zip(rows, embeddings)]
        with self._connect() as conn, conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO schema_chunks (source, chunk, metadata, embedding) VALUES %s",
                values,
                template="(%s, %s, %s, %s)",
            )
            conn.commit()
        return len(values)

    def search(self, query: str, k: int = 6) -> List[dict]:
        query_emb = self._to_pgvector(self.model.encode([query], normalize_embeddings=True)[0].tolist())
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM schema_chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (query_emb, query_emb, k),
            )
            rows = cur.fetchall()
        return [
            {"source": r[0], "chunk": r[1], "metadata": r[2], "score": float(r[3])}
            for r in rows
        ]

    @staticmethod
    def _to_pgvector(embedding: List[float]) -> str:
        # Serialize to pgvector literal, e.g. '[0.1, 0.2, ...]'
        return "[" + ", ".join(f"{x:.8f}" for x in embedding) + "]"


def flatten_schema_json(schema_json: list[dict]) -> List[Tuple[str, str, dict]]:
    """Convert a schema JSON (like DB_Schema.json) into text chunks for embedding."""
    rows: List[Tuple[str, str, dict]] = []
    for table in schema_json:
        schema = table.get("table_schema", "public")
        table_name = table.get("table_name")
        source = f"{schema}.{table_name}"
        # table summary chunk
        summary = f"Table {source}. Columns: " + ", ".join(c["column_name"] for c in table.get("columns", []))
        rows.append((source, summary, {"kind": "table", "schema": schema, "table": table_name}))
        for col in table.get("columns", []):
            col_txt = (
                f"Column {schema}.{table_name}.{col['column_name']} type {col['data_type']} "
                f"nullable={col.get('is_nullable')} default={col.get('column_default')} "
                f"desc={col.get('description')}"
            )
            rows.append(
                (
                    source,
                    col_txt,
                    {
                        "kind": "column",
                        "schema": schema,
                        "table": table_name,
                        "column": col.get("column_name"),
                    },
                )
            )
    return rows


