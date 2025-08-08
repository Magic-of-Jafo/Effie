import asyncio
from typing import Any, Dict

import httpx
import pytest

from services.gpt import generate_response


def test_generate_response_success(httpx_mock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [
                {"message": {"content": "This is a joke."}}
            ]
        },
        status_code=200,
    )

    cfg: Dict[str, Any] = {
        "openai_api_key": "test-key",
        "model_names": {"text": "gpt-4o-mini"},
    }

    async def runner() -> None:
        async with httpx.AsyncClient() as client:
            text = await generate_response("Tell me a short, one-sentence joke.", http_client=client, config=cfg)
        assert isinstance(text, str)
        assert "joke" in text.lower()

    asyncio.run(runner())


def test_generate_response_rejects_empty_prompt() -> None:
    async def runner() -> None:
        with pytest.raises(ValueError):
            await generate_response("")

    asyncio.run(runner())
