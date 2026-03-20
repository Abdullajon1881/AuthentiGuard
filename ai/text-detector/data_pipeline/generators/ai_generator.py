"""
Step 11: Generate AI text samples using GPT-4, Claude, LLaMA, and Mistral APIs
at varying temperatures and prompt configurations.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import AsyncIterator

import structlog

from ..config import (
    AI_GENERATION_CONFIGS,
    GENERATION_TOPICS,
    GENERATION_LENGTHS,
    GenerationConfig,
    INITIAL_TARGET,
    AI_RATIO,
)

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("datasets/ai-generated")


# ── Provider clients ──────────────────────────────────────────

async def _call_openai(
    config: GenerationConfig,
    prompt: str,
    system: str,
    temperature: float,
) -> str | None:
    try:
        import openai  # type: ignore
        client = openai.AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=config.max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        log.warning("openai_error", error=str(exc))
        return None


async def _call_anthropic(
    config: GenerationConfig,
    prompt: str,
    system: str,
    temperature: float,
) -> str | None:
    try:
        import anthropic  # type: ignore
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model=config.model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=config.max_tokens,
        )
        return resp.content[0].text
    except Exception as exc:
        log.warning("anthropic_error", error=str(exc))
        return None


async def _call_together(
    config: GenerationConfig,
    prompt: str,
    system: str,
    temperature: float,
) -> str | None:
    try:
        import httpx
        import os
        api_key = os.environ["TOGETHER_API_KEY"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": config.max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        log.warning("together_error", error=str(exc))
        return None


async def _generate_one(
    config: GenerationConfig,
    temperature: float,
    system: str,
    topic: str,
    length: int,
) -> str | None:
    """Call the appropriate provider for one generation."""
    prompt = config.user_prompts_template.format(topic=topic, length=length)

    if config.provider == "openai":
        return await _call_openai(config, prompt, system, temperature)
    elif config.provider == "anthropic":
        return await _call_anthropic(config, prompt, system, temperature)
    elif config.provider == "together":
        return await _call_together(config, prompt, system, temperature)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


async def _generation_stream(
    target: int,
    concurrency: int = 5,
) -> AsyncIterator[dict]:
    """
    Async generator that yields AI-generated text records.
    Distributes generation across all configured models, temperatures,
    system prompts, and topics.
    """
    semaphore = asyncio.Semaphore(concurrency)
    generated = 0

    # Build a weighted list of configs
    configs_weighted: list[GenerationConfig] = []
    for cfg in AI_GENERATION_CONFIGS:
        count = max(1, round(cfg.weight * 10))
        configs_weighted.extend([cfg] * count)

    async def _bounded_generate() -> dict | None:
        async with semaphore:
            config = random.choice(configs_weighted)
            temperature = random.choice(config.temperatures)
            system = random.choice(config.system_prompts)
            topic = random.choice(GENERATION_TOPICS)
            length = random.choice(GENERATION_LENGTHS)

            text = await _generate_one(config, temperature, system, topic, length)
            if not text or len(text.split()) < 30:
                return None

            return {
                "text": text.strip(),
                "label": 1,              # 1 = AI-generated
                "model": config.model,
                "provider": config.provider,
                "temperature": temperature,
                "topic": topic,
                "system_prompt": system,
                "word_count": len(text.split()),
            }

    tasks: list[asyncio.Task] = []
    batch_size = concurrency * 4

    while generated < target:
        # Fill task batch
        while len(tasks) < batch_size and generated + len(tasks) < target:
            tasks.append(asyncio.create_task(_bounded_generate()))

        # Wait for first completed task
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        tasks = list(pending)

        for task in done:
            result = await task
            if result:
                generated += 1
                yield result

                if generated % 500 == 0:
                    log.info("generation_progress", generated=generated, target=target)

            # Rate limit: small sleep to avoid hammering APIs
            await asyncio.sleep(0.05)


async def generate_ai_samples(
    target_total: int = int(INITIAL_TARGET * AI_RATIO),
    output_dir: Path = OUTPUT_DIR,
    concurrency: int = 5,
) -> Path:
    """
    Generate AI text samples from all configured models.

    Args:
        target_total: Total AI samples to generate.
        output_dir:   Where to write the JSONL file.
        concurrency:  Max concurrent API calls.

    Returns:
        Path to the written ai_generated.jsonl file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ai_generated.jsonl"

    log.info(
        "starting_ai_generation",
        target=target_total,
        models=[c.model for c in AI_GENERATION_CONFIGS],
        concurrency=concurrency,
    )

    total_written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        async for record in _generation_stream(target_total, concurrency):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

    log.info("generation_complete", total=total_written, path=str(out_path))
    return out_path


if __name__ == "__main__":
    asyncio.run(generate_ai_samples())
