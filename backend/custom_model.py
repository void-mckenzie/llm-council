"""Custom Model API client for making LLM requests to local/custom compatible servers."""

import httpx
from typing import List, Dict, Any, Optional
from .config import LOCAL_MODEL_BASE_URL, LOCAL_MODEL_API_KEY


async def query_custom_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via a Custom/Local OpenAI-compatible API.

    Args:
        model: Model identifier. The 'local/' prefix will be automatically stripped
               before sending to the server.
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Content-Type": "application/json",
    }
    if LOCAL_MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {LOCAL_MODEL_API_KEY}"

    # Some local servers might expect just the model name without "local/"
    # But usually it's better to pass what is requested.
    # If the user defines "local/my-model", we might want to strip "local/" 
    # if the server doesn't expect it. However, for now, let's keep it simple.
    # The config.py defines the ENDPOINT.

    payload = {
        "model": model.replace("local/", "") if model.startswith("local/") else model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                LOCAL_MODEL_BASE_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            
            # Validate response format
            if 'choices' not in data or not data['choices']:
                print(f"Unexpected response format from {model}: missing or empty 'choices'")
                return None
            
            message = data['choices'][0].get('message')
            if not message:
                print(f"Unexpected response format from {model}: missing 'message' in choice")
                return None

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying custom model {model} at {LOCAL_MODEL_BASE_URL}: {e}")
        return None
