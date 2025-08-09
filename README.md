# Text2SQL (Streamlit + pgvector + Local/Hosted LLM)

## What this does
- Vectorizes your DB schema into `pgvector` for retrieval-augmented SQL generation
- Converts natural language to SQL using a local or hosted LLM
- Executes generated SQL on your main Postgres and shows results in the UI

## Architecture
- UI: Streamlit (`app.py`)
- Vector DB: PostgreSQL with `pgvector` (`docker-compose.yml` → service `db_vector`)
- Main DB: PostgreSQL (`docker-compose.yml` → service `db`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- LLM providers:
  - Local: Ollama (`sqlcoder` recommended)
  - Hosted: Groq (free tier), Hugging Face Inference API (free tier)

## Prerequisites
- Docker Desktop running (for the Postgres and pgvector containers)
- Python 3.12
- Optional for local inference: Ollama installed (`brew install ollama`)

## Setup
1) Start databases
```bash
docker compose up -d
```

2) Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Configure environment (.env)
Create a `.env` in the project root:
```
# Choose provider: ollama | groq | hf
LLM_PROVIDER=ollama

# Ollama (local)
OLLAMA_MODEL=sqlcoder
OLLAMA_BASE_URL=http://localhost:11434

# Groq (hosted)
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_key
# GROQ_MODEL=llama-3.1-8b-instant
# GROQ_BASE_URL=https://api.groq.com/openai/v1

# Hugging Face Inference API (hosted)
# LLM_PROVIDER=hf
# HF_API_KEY=your_key
# HF_MODEL=google/gemma-2-2b-it

# Optional DB overrides (instead of editing config.py)
# MAIN_DB_HOST=localhost
# MAIN_DB_PORT=5432
# MAIN_DB_NAME=mydatabase
# MAIN_DB_USER=myuser
# MAIN_DB_PASSWORD=mypassword
# VECTOR_DB_HOST=localhost
# VECTOR_DB_PORT=5433
# VECTOR_DB_NAME=vectordb
# VECTOR_DB_USER=vectoruser
# VECTOR_DB_PASSWORD=vectorpass
```

4) Optional: Local model with Ollama
```bash
brew install ollama
ollama serve
ollama pull sqlcoder
```

5) Run the app
```bash
streamlit run app.py
```

## Using the app
- Sidebar → Schema Vectorization:
  - Upload a schema JSON (see `DB_Schema.json`) to index into `pgvector`, or
  - Click “Fetch current DB schema (public)” to introspect and index the live DB
- Main panel:
  - Ask a question in natural language
  - See the schema context retrieved from `pgvector`
  - See the generated SQL (only `SELECT` is executed by default for safety)
  - View results in a table

## Troubleshooting
- Docker not running: Start Docker Desktop, then `docker compose up -d` again
- pgvector index error:
  - The app now uses `vector_cosine_ops` and falls back if index creation fails
  - Manually create if needed:
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE INDEX IF NOT EXISTS idx_schema_chunks_embedding
    ON schema_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    -- Or use L2 if cosine ops are unavailable:
    -- ON schema_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
    ANALYZE schema_chunks;
    ```
- NumPy/Torch compatibility (Python 3.12):
  - Dependencies are pinned for 3.12 in `requirements.txt` (NumPy 2.x, Torch 2.4.1, Transformers 4.44)
  - Reinstall:
    ```bash
    source .venv/bin/activate
    pip install --force-reinstall -r requirements.txt
    ```
- LLM provider issues:
  - Local: ensure `ollama serve` is running and a model is pulled
  - Groq: set `LLM_PROVIDER=groq` and `GROQ_API_KEY`
  - HF: set `LLM_PROVIDER=hf` and `HF_API_KEY`

## Security & production notes
- Add auth, logging, and SQL safety checks before production
- Consider server-side rate limiting and audit logs
- Restrict DB roles to read-only for the app connection
