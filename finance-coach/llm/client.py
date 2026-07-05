"""Groq-primary / Gemini-fallback LLM client."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Always load finance-coach/.env (not cwd-dependent)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH, override=True)

_PLACEHOLDERS = {
    "",
    "your_groq_api_key_here",
    "your_google_api_key_here",
}


def _env_key(name: str) -> str | None:
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not value or value in _PLACEHOLDERS:
        return None
    return value


def llm_available() -> bool:
    return bool(_env_key("GROQ_API_KEY") or _env_key("GOOGLE_API_KEY"))


def _groq_model():
    api_key = _env_key("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
        )
    except Exception:
        return None


def _gemini_model():
    api_key = _env_key("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.3,
        )
    except Exception:
        return None


def complete(system: str, user: str) -> str:
    """Call Groq first; fall back to Gemini on failure. Returns empty string if no keys."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=system), HumanMessage(content=user)]

    groq = _groq_model()
    if groq is not None:
        try:
            response = groq.invoke(messages)
            content = getattr(response, "content", str(response))
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            return str(content).strip()
        except Exception:
            pass

    gemini = _gemini_model()
    if gemini is not None:
        try:
            response = gemini.invoke(messages)
            content = getattr(response, "content", str(response))
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            return str(content).strip()
        except Exception:
            pass

    return ""
