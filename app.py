import json
from typing import List

import streamlit as st
from dotenv import load_dotenv

from vector_store import VectorStore, flatten_schema_json
from schema_utils import fetch_schema_from_postgres
from db import SQLDatabase
from llm import LLM


load_dotenv()
st.set_page_config(page_title="Text2SQL", layout="wide")


@st.cache_resource
def get_services():
    return VectorStore(), SQLDatabase(), LLM()


def sidebar_ingest_ui(vs: VectorStore) -> None:
    st.sidebar.header("Schema Vectorization")
    uploaded = st.sidebar.file_uploader("Upload DB schema JSON", type=["json"], accept_multiple_files=False)
    reset = st.sidebar.button("Clear Indexed Schema", use_container_width=True)
    if reset:
        vs.clear()
        st.sidebar.success("Vector store cleared.")
    if uploaded:
        schema = json.load(uploaded)
        rows = flatten_schema_json(schema)
        vs.setup()
        n = vs.upsert_chunks(rows)
        st.sidebar.success(f"Indexed {n} chunks from schema.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Fetch current DB schema (public)"):
        schema = fetch_schema_from_postgres()
        rows = flatten_schema_json(schema)
        vs.setup()
        n = vs.upsert_chunks(rows)
        st.sidebar.success(f"Indexed {n} chunks from live DB schema.")


def main_query_ui(vs: VectorStore, db: SQLDatabase, llm: LLM) -> None:
    st.title("Text → SQL for PostgreSQL")
    question = st.text_input("Ask a question about your data")
    top_k = st.slider("Schema context size", min_value=3, max_value=15, value=8)
    run = st.button("Generate & Run", type="primary")

    if run and question:
        vs.setup()
        hits = vs.search(question, k=top_k)
        context_lines: List[str] = [h["chunk"] for h in hits]
        with st.expander("Schema context"):
            for h in hits:
                st.write(f"[{h['score']:.2f}] {h['chunk']}")

        sql = llm.generate_sql(question, context_lines)
        st.code(sql, language="sql")

        if not sql.strip().lower().startswith("select"):
            st.warning("Generated SQL is not a SELECT. Skipping execution for safety.")
            return

        try:
            df = db.run_query(sql)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            return
        st.dataframe(df, use_container_width=True)


def main():
    vs, db, llm = get_services()
    sidebar_ingest_ui(vs)
    main_query_ui(vs, db, llm)


if __name__ == "__main__":
    main()


