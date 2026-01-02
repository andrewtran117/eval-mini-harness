"""
Client wrapper for OpenAI-compatible endpoints (vLLM).

Handles request/response with metrics collection and retries.
"""

import os
import time
from typing import Dict, Any, Tuple
from openai import OpenAI


class ClientError(Exception):
    """Raised when client operations fail."""
    pass


_CLIENT_CACHE: Dict[Tuple[str, str], OpenAI] = {}


def get_client_params(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Resolve base_url and api_key from config or environment.
    Defaults match vLLM's OpenAI-compatible server (no auth).
    """
    client_cfg = (config.get("client") or {}) if config else {}
    base_url = (
        client_cfg.get("base_url")
        or (config.get("base_url") if config else None)
        or os.environ.get("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    api_key = (
        client_cfg.get("api_key")
        or (config.get("api_key") if config else None)
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    return base_url, api_key


def _get_client(base_url: str, api_key: str) -> OpenAI:
    key = (base_url, api_key)
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENT_CACHE[key]


def get_openai_client(config: Dict[str, Any]) -> OpenAI:
    """Shared client constructor with caching."""
    base_url, api_key = get_client_params(config)
    return _get_client(base_url, api_key)


def run_inference(
    config: Dict[str, Any],
    prompt: str,
    max_retries: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Send a prompt to an OpenAI-compatible endpoint (e.g., vLLM) and get response with metrics.

    Args:
        config: Configuration dictionary
        prompt: Prompt text to send
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (response_text, metrics_dict)

    Raises:
        ClientError: If request fails after all retries
    """
    model = config["model"]
    temperature = config["sampling"].get("temperature", 0.0)
    seed = config["run"].get("seed")
    max_tokens = config["sampling"].get("max_tokens", 512)
    base_url, api_key = get_client_params(config)
    client = _get_client(base_url, api_key)

    last_error = None
    for attempt in range(max_retries):
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

            end_time = time.time()
            latency_s = end_time - start_time

            choice = response.choices[0].message
            response_text = choice.content if choice else ""
            usage = response.usage or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            metrics = {
                "latency_s": latency_s,
                "ttft_ms": None,
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
                "tokens_total": None,
            }

            if metrics["tokens_in"] and metrics["tokens_out"]:
                metrics["tokens_total"] = metrics["tokens_in"] + metrics["tokens_out"]

            return response_text, metrics

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise ClientError(f"Request failed after {max_retries} attempts: {e}")

    raise ClientError(f"Request failed: {last_error}")


def test_connection(config: Dict[str, Any]) -> bool:
    """
    Test connection to the OpenAI-compatible endpoint by listing models or making a quick request.
    """
    base_url, api_key = get_client_params(config)
    client = _get_client(base_url, api_key)
    try:
        models = client.models.list()
        return bool(getattr(models, "data", None))
    except Exception:
        try:
            _ = run_inference(config, "ping", max_retries=1)
            return True
        except Exception:
            return False


def get_model_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about available models from the OpenAI-compatible endpoint.
    """
    base_url, api_key = get_client_params(config)
    client = _get_client(base_url, api_key)
    try:
        models = client.models.list()
        names = [m.id for m in getattr(models, "data", [])]
        return {"available_models": names, "count": len(names)}
    except Exception:
        return {"available_models": [], "count": 0}
