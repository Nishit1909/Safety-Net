"""Single gateway to every model: Groq -> retry -> Gemini fallback, validated by Pydantic.

Set SAFETYNET_MOCK=1 to run without any API (used by unit tests).
"""
import json
import os
import threading
import time

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b"}
GEMINI_MODEL = "gemini-2.5-flash"

# Cap concurrent calls so parallel branches don't blow the free-tier RPM limit.
_SEM = threading.Semaphore(int(os.getenv("SAFETYNET_MAX_CONCURRENCY", "3")))

# Run trace: every call is logged here (provider, model, tokens, latency).
CALL_LOG: list[dict] = []
_LOG_LOCK = threading.Lock()

_groq = None
_gemini = None


def _groq_client():
    global _groq
    if _groq is None:
        from groq import Groq
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


def _gemini_client():
    global _gemini
    if _gemini is None and os.getenv("GEMINI_API_KEY"):
        from google import genai
        _gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini


def _call_groq(prompt, model, temperature):
    resp = _groq_client().chat.completions.create(
        model=MODELS[model],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    return resp.choices[0].message.content, tokens


def _call_gemini(prompt, temperature):
    resp = _gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": temperature},
    )
    if not resp.text:
        reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
        raise RuntimeError(f"Gemini returned no text (finish_reason={reason})")
    meta = resp.usage_metadata
    return resp.text, (meta.total_token_count or 0) if meta else 0


_MOCK_HANDLER = None


def set_mock_handler(fn):
    """fn(prompt, schema, agent) -> dict. Used only in tests."""
    global _MOCK_HANDLER
    _MOCK_HANDLER = fn


def call_llm(prompt: str, schema: type[BaseModel], model: str = "small",
             temperature: float = 0.0, agent: str = "") -> BaseModel:
    if os.getenv("SAFETYNET_MOCK") == "1":
        return schema.model_validate(_MOCK_HANDLER(prompt, schema, agent))

    full_prompt = (prompt + "\n\nRespond ONLY with one JSON object matching this JSON schema:\n"
                   + json.dumps(schema.model_json_schema()))

    # Bad JSON at temperature 0 tends to repeat, so retries add a little temperature.
    attempts = [("groq", lambda: _call_groq(full_prompt, model, temperature)),
                ("groq-retry", lambda: _call_groq(full_prompt, model, temperature + 0.3)),
                ("groq-retry2", lambda: _call_groq(full_prompt, model, temperature + 0.6))]
    if _gemini_client():
        attempts.append(("gemini", lambda: _call_gemini(full_prompt, temperature)))
        attempts.append(("gemini-retry", lambda: (time.sleep(10), _call_gemini(full_prompt, temperature))[1]))

    errors = []
    with _SEM:
        for i, (provider, fn) in enumerate(attempts):
            t0 = time.time()
            try:
                raw, tokens = fn()
                out = schema.model_validate_json(raw)
                with _LOG_LOCK:
                    CALL_LOG.append({"agent": agent, "provider": provider, "model": model,
                                     "tokens": tokens, "latency_s": round(time.time() - t0, 2)})
                return out
            except (ValidationError, json.JSONDecodeError) as e:
                errors.append(f"{provider}: bad JSON -> {str(e)[:200]}")
            except Exception as e:  # rate limit, network
                errors.append(f"{provider}: {type(e).__name__} -> {str(e)[:200]}")
                if "rate" in str(e).lower() or "503" in str(e):
                    time.sleep(8)
    raise RuntimeError("All providers failed:\n" + "\n".join(errors))
