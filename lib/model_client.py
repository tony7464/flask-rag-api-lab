import requests

DEFAULT_MODEL_NAME = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECONDS = 60


class ModelClientError(Exception):
    """Raised when the local model service cannot return a usable answer."""


def generate_answer(prompt, model_name=DEFAULT_MODEL_NAME, base_url=DEFAULT_BASE_URL):
    """Send the prompt to a local Ollama model and return the generated answer.

    Args:
        prompt: The full prompt text to send to the model.
        model_name: The Ollama model to use.
        base_url: The base URL of the Ollama service.

    Returns:
        The model's answer text with surrounding whitespace removed.

    Raises:
        ModelClientError: If the prompt is blank, the request fails, the
            service returns invalid JSON, or the response field is missing
            or blank.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ModelClientError("Prompt must be a non-empty string.")

    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ModelClientError(f"Model request failed: {exc}") from exc

    # requests' JSONDecodeError subclasses both RequestException and ValueError,
    # so decoding must be handled separately to report it accurately.
    try:
        data = response.json()
    except ValueError as exc:
        raise ModelClientError("Model service returned invalid JSON.") from exc

    answer = data.get("response") if isinstance(data, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ModelClientError("Model service returned no usable response.")

    return answer.strip()
