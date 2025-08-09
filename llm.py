from typing import List
import requests
import os
import logging
import json
from pathlib import Path


class LLM:
    """LLM client supporting multiple providers (ollama, groq, hf). Default: ollama.

    Providers:
      - ollama (local): OLLAMA_MODEL, OLLAMA_BASE_URL
      - groq (hosted, free tier): GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
      - hf (Hugging Face Inference API): HF_API_KEY, HF_MODEL
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.model = model or os.getenv("OLLAMA_MODEL", "sqlcoder")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Groq config
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        # Hugging Face config
        self.hf_api_key = os.getenv("HF_API_KEY")
        self.hf_model = os.getenv("HF_MODEL", "google/gemma-2-2b-it")

        # Logging config
        self.log_enabled = os.getenv("LLM_LOG_ENABLED", "1") not in {"0", "false", "False"}
        self.log_file = os.getenv("LLM_LOG_FILE", "logs/llm.log")
        try:
            self.log_max_chars = int(os.getenv("LLM_LOG_MAX_CHARS", "2000"))
        except ValueError:
            self.log_max_chars = 2000
        self._setup_logger()

    def generate_sql(self, question: str, schema_context: List[str]) -> str:
        prompt = self._build_prompt(question, schema_context)
        if self.provider == "groq":
            return self._generate_sql_groq(prompt)
        if self.provider == "hf":
            return self._generate_sql_hf(prompt)
        return self._generate_sql_ollama(prompt)

    def _generate_sql_ollama(self, prompt: str) -> str:
        self._log_event("request", {"provider": "ollama", "model": self.model, "prompt": prompt})
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        self._log_event("response", {"provider": "ollama", "model": self.model, "response": text})
        return self._postprocess_sql(text)

    def _generate_sql_groq(self, prompt: str) -> str:
        self._log_event("request", {"provider": "groq", "model": self.groq_model, "prompt": prompt})
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": "Return only valid PostgreSQL SQL without fences."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        resp = requests.post(f"{self.groq_base}/chat/completions", json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        self._log_event("response", {"provider": "groq", "model": self.groq_model, "response": text})
        return self._postprocess_sql(text)

    def _generate_sql_hf(self, prompt: str) -> str:
        self._log_event("request", {"provider": "hf", "model": self.hf_model, "prompt": prompt})
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.2, "max_new_tokens": 512},
            "options": {"wait_for_model": True},
        }
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{self.hf_model}", json=payload, headers=headers, timeout=180
        )
        resp.raise_for_status()
        data = resp.json()
        # Responses can be either a list of dicts or a dict with 'generated_text'
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "") or data[0].get("summary_text", "") or ""
        else:
            text = data.get("generated_text", "") if isinstance(data, dict) else ""
        if not text and isinstance(data, list) and data and "generated_text" not in data[0]:
            # Some models return {'generated_text': ...} nested in 'content'
            text = str(data)
        self._log_event("response", {"provider": "hf", "model": self.hf_model, "response": text})
        return self._postprocess_sql(text)

    @staticmethod
    def _build_prompt(question: str, schema_context: List[str]) -> str:
        context = "\n".join(schema_context)
        return (
            "You are a helpful assistant that writes syntactically correct PostgreSQL SQL based on the provided database schema.\n"
            "- Use only tables, columns, and relations that appear in the context.\n"
            "- Prefer simple SELECTs. If ambiguous, make reasonable assumptions.\n"
            "- Return ONLY the SQL query. Do not include explanations or markdown fences.\n\n"
            f"Schema Context:\n{context}\n\n"
            f"User Question: {question}\n\n"
            "SQL:"
        )

    @staticmethod
    def _postprocess_sql(text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        # Remove backtick or triple-backtick fences if present
        if t.startswith("```"):
            t = t.strip("`")
            # Strip 'sql' language tag if present
            if t.lower().startswith("sql"):
                t = t[3:]
        # Remove stray code fences
        t = t.replace("```sql", "").replace("```", "").strip()
        return t

    # ----------------- Logging helpers -----------------
    _logger_initialized = False

    def _setup_logger(self) -> None:
        if not self.log_enabled:
            return
        if LLM._logger_initialized:
            return
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("llm")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        LLM._logger_initialized = True

    def _log_event(self, kind: str, payload: dict) -> None:
        if not self.log_enabled:
            return
        try:
            truncated = {}
            for k, v in payload.items():
                if isinstance(v, str) and len(v) > self.log_max_chars:
                    truncated[k] = v[: self.log_max_chars] + "...<truncated>"
                else:
                    truncated[k] = v
            logger = logging.getLogger("llm")
            logger.info(json.dumps({"event": kind, **truncated}))
        except Exception:
            # Never crash the app due to logging
            pass


