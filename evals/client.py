"""
Client wrapper for vLLM (OpenAI-compatible) endpoints.

Handles request/response with metrics collection, retries, and timeouts.
"""

import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI, OpenAIError
import httpx


class ClientError(Exception):
    """Raised when client operations fail."""
    pass


def create_client(config: Dict[str, Any]) -> OpenAI:
    """
    Create OpenAI client pointed at vLLM endpoint.

    Args:
        config: Configuration dictionary with endpoint settings

    Returns:
        OpenAI client instance
    """
    base_url = config["endpoint"]["base_url"]
    api_key = config["endpoint"].get("api_key", "dummy")

    # Create client with longer timeout for LLM inference
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(120.0, connect=10.0),  # 120s inference, 10s connect
    )

    return client


def run_inference(
    client: OpenAI,
    config: Dict[str, Any],
    prompt: str,
    max_retries: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Send a prompt to vLLM and get response with metrics.

    Args:
        client: OpenAI client instance
        config: Configuration dictionary
        prompt: Prompt text to send
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (response_text, metrics_dict)

    Raises:
        ClientError: If request fails after all retries
    """
    model = config["model"]
    sampling = config["sampling"]
    seed = config["run"].get("seed")
    stream = sampling.get("stream", False)

    # Build request parameters
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": sampling.get("temperature", 0.0),
        "max_tokens": sampling.get("max_tokens", 512),
        "stream": stream,
    }

    # Add seed if specified
    if seed is not None:
        params["seed"] = seed

    # Retry logic with exponential backoff
    last_error = None
    for attempt in range(max_retries):
        try:
            if stream:
                return _run_streaming_inference(client, params)
            else:
                return _run_nonstreaming_inference(client, params)

        except OpenAIError as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise ClientError(f"Request failed after {max_retries} attempts: {e}")

        except Exception as e:
            raise ClientError(f"Unexpected error during inference: {e}")

    raise ClientError(f"Request failed: {last_error}")


def _run_nonstreaming_inference(
    client: OpenAI,
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Run non-streaming inference and collect metrics.

    Args:
        client: OpenAI client
        params: Request parameters

    Returns:
        Tuple of (response_text, metrics_dict)
    """
    start_time = time.time()

    response = client.chat.completions.create(**params)

    end_time = time.time()
    latency_s = end_time - start_time

    # Extract response text
    response_text = response.choices[0].message.content or ""

    # Collect metrics
    metrics = {
        "latency_s": latency_s,
        "ttft_ms": None,  # Not available in non-streaming
        "tokens_in": response.usage.prompt_tokens if response.usage else None,
        "tokens_out": response.usage.completion_tokens if response.usage else None,
        "tokens_total": response.usage.total_tokens if response.usage else None,
    }

    return response_text, metrics


def _run_streaming_inference(
    client: OpenAI,
    params: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Run streaming inference and collect metrics including TTFT.

    Args:
        client: OpenAI client
        params: Request parameters

    Returns:
        Tuple of (response_text, metrics_dict)
    """
    start_time = time.time()
    first_token_time = None
    chunks = []

    stream = client.chat.completions.create(**params)

    for chunk in stream:
        # Record time to first token
        if first_token_time is None and chunk.choices:
            first_token_time = time.time()

        # Collect chunk content
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    end_time = time.time()

    # Calculate metrics
    latency_s = end_time - start_time
    ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else None

    # Combine response
    response_text = "".join(chunks)

    metrics = {
        "latency_s": latency_s,
        "ttft_ms": ttft_ms,
        "tokens_in": None,  # Not easily available in streaming
        "tokens_out": None,  # Could estimate with tokenizer
        "tokens_total": None,
    }

    return response_text, metrics


def test_connection(client: OpenAI, config: Dict[str, Any]) -> bool:
    """
    Test connection to vLLM endpoint with a simple request.

    Args:
        client: OpenAI client
        config: Configuration dictionary

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to list models (vLLM supports this endpoint)
        models = client.models.list()
        return True
    except Exception:
        # If models.list fails, try a simple inference
        try:
            response_text, metrics = run_inference(
                client,
                config,
                "Hello",
                max_retries=1
            )
            return True
        except Exception:
            return False


def get_model_info(client: OpenAI) -> Optional[Dict[str, Any]]:
    """
    Get information about available models from vLLM.

    Args:
        client: OpenAI client

    Returns:
        Dictionary with model info, or None if unavailable
    """
    try:
        models = client.models.list()
        if models.data:
            return {
                "available_models": [m.id for m in models.data],
                "count": len(models.data),
            }
        return None
    except Exception:
        return None


async def run_inference_async(
    client: OpenAI,
    config: Dict[str, Any],
    prompt: str,
    max_retries: int = 3
) -> Tuple[str, Dict[str, Any]]:
    """
    Async version of run_inference (for future parallel execution).

    Note: This is a placeholder. Full async implementation would require
    the async OpenAI client.

    Args:
        client: OpenAI client
        config: Configuration dictionary
        prompt: Prompt text
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (response_text, metrics_dict)
    """
    # For now, just call the sync version
    # TODO: Implement proper async with AsyncOpenAI
    return run_inference(client, config, prompt, max_retries)
