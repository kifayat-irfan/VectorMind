"""
LLM client for answer generation. Ollama by default, matching the "works
offline, no cloud APIs" requirement — talks to a local Ollama server over
plain HTTP, no extra SDK.

H5 hardening: every HTTP call carries an explicit timeout (a hung Ollama
used to hang the request forever), and the Groq client retries transient
429/5xx failures with exponential backoff instead of killing the query.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp

# Generous but finite: local 8B models can take a while on long contexts.
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=30)
# Shorter for the hosted API — it should respond quickly or not at all.
GROQ_TIMEOUT = aiohttp.ClientTimeout(total=120, connect=30)

MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def _post_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    json: dict,
    headers: Optional[dict] = None,
    timeout: aiohttp.ClientTimeout,
    retries: int = MAX_RETRIES,
) -> dict:
    """POST with exponential backoff on transient failures (429/5xx)."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            async with session.post(url, json=json, headers=headers, timeout=timeout) as resp:
                if resp.status in RETRYABLE_STATUS and attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                if resp.status != 200:
                    body = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status, message=body[:300]
                    )
                return await resp.json()
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


class OllamaClient:
    """
    Args:
        base_url: Ollama server URL (default assumes a locally-running instance).
        model: Model tag, e.g. "llama3.1:8b". Must already be pulled (`ollama pull llama3.1:8b`).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_get(self) -> aiohttp.ClientSession:
        # L4: one shared session per client instead of a new one per call.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        Args:
            prompt: User-turn content.
            system: System prompt.
            temperature: Sampling temperature; low by default since this is
                a factual QA context where creativity is undesirable.

        Returns:
            The model's response text.

        Raises:
            aiohttp.ClientError / asyncio.TimeoutError: on connection failure
                or timeout (e.g. Ollama not running).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        session = await self._session_get()
        data = await _post_with_retry(
            session, f"{self.base_url}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT
        )
        return data.get("response", "")


class SenseNovaLLMClient:
    """
    Public-mode LLM client — calls SenseNova's hosted chat completions API
    (OpenAI-compatible) instead of a local Ollama instance. This replaced
    Groq as the hosted fallback per the owner's decision (2026-10-23).
    Local mode (OllamaClient above) stays the default everywhere else.

    Auth (in order of preference):
      1. ``api_key`` passed explicitly — env SENSENOVA_API_KEY (or
         RAG_SENSENOVA_API_KEY), or Streamlit secrets on a deployed demo.
         NEVER hardcode a real key here or in any config/prompt.
      2. No key: uses the stored ``custom.sensenova`` credential through the
         same surrogate mechanism as the project's SenseNova connector CLI —
         zero-config on machines where it's connected. No raw key is ever
         read, printed, or persisted by this client.

    Args:
        api_key: SenseNova API key (optional — see auth above).
        model: SenseNova-hosted model id, e.g. "sensenova-6.8-flash-lite".
    """

    def __init__(self, api_key: str = "", model: str = "sensenova-6.8-flash-lite"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://token.sensenova.ai/v1/chat/completions"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _generate_via_stored_credential(endpoint: str, payload: dict) -> str:
        """Sync POST using the stored custom.sensenova credential (surrogate
        auth, same mechanism as the project's SenseNova connector CLI).
        Runs in a worker thread via asyncio.to_thread. Retries transient
        failures with exponential backoff."""
        import json as _json
        import sys as _sys
        import time as _time
        import urllib.request as _urlreq

        _sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
        from dynamic_credentials import (  # noqa: E402
            DynamicCredentialError,
            add_surrogate_to_request,
            read_json_response,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            request = _urlreq.Request(
                endpoint,
                data=_json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                add_surrogate_to_request(
                    request, "custom.sensenova", allowed_hosts=("token.sensenova.ai",)
                )
            except DynamicCredentialError as exc:
                raise RuntimeError(f"SenseNova credential exchange failed: {exc}") from exc
            try:
                with _urlreq.urlopen(request, timeout=120) as resp:
                    body = read_json_response(resp)
                return body["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - classified below
                status = getattr(exc, "code", None)
                last_exc = exc
                if status in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    _time.sleep(2**attempt)
                    continue
                if status in (401, 403):
                    raise RuntimeError(
                        "SenseNova rejected the stored credential (auth). "
                        "Reconnect the SenseNova connector, or set SENSENOVA_API_KEY."
                    ) from exc
                raise
        assert last_exc is not None
        raise last_exc

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "temperature": temperature}

        if self.api_key:
            # Explicit key path (deployed demo / Streamlit secrets).
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            session = await self._session_get()
            data = await _post_with_retry(
                session, self.endpoint, json=payload, headers=headers, timeout=GROQ_TIMEOUT
            )
            return data["choices"][0]["message"]["content"]

        # Stored-credential path — no raw key involved.
        return await asyncio.to_thread(
            self._generate_via_stored_credential, self.endpoint, payload
        )


class GroqLLMClient:
    """Legacy hosted client (Groq). Kept for backwards compatibility —
    SenseNovaLLMClient is now the default Public-mode client."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise ValueError(
                "No Groq API key configured. Set GROQ_API_KEY (or RAG_GROQ_API_KEY) "
                "to use Public mode — get a free key at https://console.groq.com/keys"
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        session = await self._session_get()
        data = await _post_with_retry(
            session, self.endpoint, json=payload, headers=headers, timeout=GROQ_TIMEOUT
        )
        return data["choices"][0]["message"]["content"]
