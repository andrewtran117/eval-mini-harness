"""
Client wrapper for Ollama.

Handles request/response with metrics collection and retries.
"""

import time
from typing import Dict, Any, Tuple
import ollama


class ClientError(Exception):
    """Raised when client operations fail."""
    pass


def run_inference(
    config: Dict[str, Any],
    prompt: str,
    max_retries: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Send a prompt to Ollama and get response with metrics.

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

    # Retry logic with exponential backoff
    last_error = None
    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # Call Ollama
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": config["sampling"].get("max_tokens", 512),
                    "seed": config["run"].get("seed"),
                }
            )

            end_time = time.time()
            latency_s = end_time - start_time

            # Extract response text
            response_text = response['message']['content']

            # Collect metrics
            metrics = {
                "latency_s": latency_s,
                "ttft_ms": None,  # Ollama doesn't expose TTFT in non-streaming
                "tokens_in": response.get('prompt_eval_count'),
                "tokens_out": response.get('eval_count'),
                "tokens_total": None,
            }

            if metrics["tokens_in"] and metrics["tokens_out"]:
                metrics["tokens_total"] = metrics["tokens_in"] + metrics["tokens_out"]

            return response_text, metrics

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise ClientError(f"Request failed after {max_retries} attempts: {e}")

    raise ClientError(f"Request failed: {last_error}")


def test_connection(config: Dict[str, Any]) -> bool:
    """
    Test connection to Ollama with a simple request.

    Args:
        config: Configuration dictionary

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to list models
        models = ollama.list()
        return True
    except Exception:
        # If list fails, try a simple inference
        try:
            response_text, metrics = run_inference(config, "Hello", max_retries=1)
            return True
        except Exception:
            return False


def get_model_info() -> Dict[str, Any]:
    """
    Get information about available models from Ollama.

    Returns:
        Dictionary with model info, or None if unavailable
    """
    try:
        models = ollama.list()
        if models.get('models'):
            return {
                "available_models": [m['name'] for m in models['models']],
                "count": len(models['models']),
            }
        return {"available_models": [], "count": 0}
    except Exception:
        return {"available_models": [], "count": 0}
