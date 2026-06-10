"""Latency benchmark for each pipeline stage.

Measures, against live APIs:
  1. STT: ElevenLabs speech-to-text on a sample audio file
  2. LLM: time-to-first-token (streaming) and time-to-full-response
  3. TTS: time-to-first-audio-chunk for configured model vs eleven_flash_v2_5

Run from corindagpt/:  ../.venv/Scripts/python -m scripts.latency_bench
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.initialization import load_config  # noqa: E402

SAMPLE_AUDIO = Path(__file__).resolve().parents[2] / "smoke_tts_output.mp3"
LLM_PROMPT = "You are a theatrical AI assistant. Reply in one short sentence: greet the audience."
TTS_TEXT = "Ladies and gentlemen, prepare to be amazed by what comes next."
RUNS = 3


def med(values: list[float]) -> int:
    s = sorted(values)
    return int(s[len(s) // 2])


async def bench_stt(cfg: dict) -> None:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=cfg["elevenlabs_api_key"])
    data = SAMPLE_AUDIO.read_bytes()
    times = []
    for _ in range(RUNS):
        t0 = time.monotonic()
        result = await asyncio.to_thread(
            client.speech_to_text.convert, file=data, model_id="scribe_v1"
        )
        times.append((time.monotonic() - t0) * 1000)
    print(f"STT  scribe_v1 ({len(data)} bytes audio): median {med(times)} ms  {[int(t) for t in times]}")
    print(f"     transcript: {getattr(result, 'text', '?')!r}")


async def bench_llm(cfg: dict) -> None:
    api_key = cfg["openai_api_key"]
    model = (cfg.get("model_names") or {}).get("text", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": LLM_PROMPT}],
        "max_tokens": 64,
        "temperature": 0.7,
        "stream": True,
    }
    ttfts, totals = [], []
    async with httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=30.0) as client:
        for _ in range(RUNS):
            t0 = time.monotonic()
            ttft = None
            async with client.stream("POST", "/chat/completions", headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    delta = (json.loads(line[6:])["choices"][0].get("delta") or {})
                    if ttft is None and delta.get("content"):
                        ttft = (time.monotonic() - t0) * 1000
            totals.append((time.monotonic() - t0) * 1000)
            ttfts.append(ttft or totals[-1])
    print(f"LLM  {model}: first-token median {med(ttfts)} ms, full-response median {med(totals)} ms")


async def bench_tts(cfg: dict, model_id: str) -> None:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=cfg["elevenlabs_api_key"])
    voice_id = cfg["tts"]["elevenlabs"]["voice_id"]
    ttfbs = []
    for _ in range(RUNS):
        def _first_chunk_ms() -> float:
            t0 = time.monotonic()
            stream = client.text_to_speech.stream(
                text=TTS_TEXT, voice_id=voice_id, model_id=model_id, output_format="pcm_16000"
            )
            for chunk in stream:
                if chunk:
                    elapsed = (time.monotonic() - t0) * 1000
                    # drain remainder without timing
                    for _ in stream:
                        pass
                    return elapsed
            return (time.monotonic() - t0) * 1000

        ttfbs.append(await asyncio.to_thread(_first_chunk_ms))
    print(f"TTS  {model_id}: first-audio-chunk median {med(ttfbs)} ms  {[int(t) for t in ttfbs]}")


async def main() -> None:
    cfg = load_config()
    print(f"--- Latency benchmark ({RUNS} runs each, medians) ---")
    await bench_stt(cfg)
    await bench_llm(cfg)
    await bench_tts(cfg, cfg["tts"]["elevenlabs"]["model_id"])
    await bench_tts(cfg, "eleven_flash_v2_5")


if __name__ == "__main__":
    asyncio.run(main())
