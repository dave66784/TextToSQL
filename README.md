# Text2SQL (Streamlit + pgvector + Ollama)

## What this does
- Vectorizes a DB schema JSON into `pgvector`
- Uses a local LLM (Ollama) to convert natural language into SQL
- Executes SQL against your main Postgres database
- Streamlit UI: upload schema JSON, ask questions, see generated SQL and results

## Prereqs
- Docker (for Postgres and pgvector containers)
- Python 3.10+
- Ollama running locally: `brew install ollama && ollama serve`
- Pull a free model: `ollama pull sqlcoder` (good for Text2SQL) or `ollama pull llama3.1`

## Configure
Edit `config.py` for your two DBs:
- `DB_STANDARD`: your main data Postgres
- `DB_VECTOR`: your pgvector Postgres (from docker compose)

## Run DBs
```bash
cd /Users/saurabhdave/Developer/WorkSpace/Text2SQL
docker compose up -d
```

## Install app
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure environment
- Create a `.env` file in the project root (same folder as `app.py`). Example:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=sqlcoder
OLLAMA_BASE_URL=http://localhost:11434

# Or Groq
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_key
# GROQ_MODEL=llama-3.1-8b-instant

# Or Hugging Face
# LLM_PROVIDER=hf
# HF_API_KEY=your_key
# HF_MODEL=google/gemma-2-2b-it
```
The app auto-loads `.env`.

## Start Ollama (local LLM)
```bash
# Install once
brew install ollama
# Start server
ollama serve
# In another terminal, pull a model
ollama pull sqlcoder
```
Optionally use `OLLAMA_MODEL` env var to switch models (e.g. `llama3.1`).

## Start app
```bash
streamlit run app.py
```

## Use
- Sidebar: upload `DB_Schema.json` (example provided) to index
- Main: ask a natural language question → shows SQL and results

## Notes
- Embeddings: `all-MiniLM-L6-v2` (384 dims). Table `schema_chunks` auto-created and indexed.
- Only `SELECT` queries are executed by default for safety.
- Change model via env vars: `OLLAMA_MODEL` and `OLLAMA_BASE_URL`.
- For production, add auth, logging, rate limiting, and SQL safety checks.
