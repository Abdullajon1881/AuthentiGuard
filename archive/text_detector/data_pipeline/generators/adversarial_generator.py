"""
Adversarial sample generation: takes AI-generated text and applies evasion
transforms — paraphrasing, back-translation, grammar correction, mixed content.
These samples train Layer 4 (Adversarial Detector) of the ensemble.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("datasets/adversarial")

AttackType = Literal["paraphrase", "backtranslation", "grammar_correction", "mixed"]

# Languages for back-translation (pivot languages)
PIVOT_LANGUAGES = ["fr", "de", "es", "zh", "ja", "ar", "ru", "pt"]


# ── Attack implementations ────────────────────────────────────

async def _paraphrase(text: str) -> str | None:
    """
    Paraphrase using GPT-4 with explicit instruction to rewrite completely.
    This is the most common evasion technique — tests Layer 4 robustness.
    """
    try:
        import openai  # type: ignore
        client = openai.AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a paraphrasing tool. Rewrite the following text "
                        "completely in your own words. Change sentence structure, "
                        "vocabulary, and phrasing while preserving the meaning. "
                        "Do not add new information. Output only the rewritten text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        log.warning("paraphrase_error", error=str(exc))
        return None


async def _backtranslate(text: str, pivot_lang: str) -> str | None:
    """
    Back-translation: EN → pivot → EN using Google Translate API.
    Introduces natural variation that evades surface-level detectors.
    """
    try:
        from google.cloud import translate_v2 as translate  # type: ignore

        client = translate.Client()

        # Forward: EN → pivot
        forward = client.translate(text, target_language=pivot_lang)
        intermediate = forward["translatedText"]

        # Backward: pivot → EN
        backward = client.translate(intermediate, target_language="en")
        return backward["translatedText"]

    except ImportError:
        # Fallback: use Helsinki-NLP models locally (slower but no API needed)
        return await _backtranslate_local(text, pivot_lang)
    except Exception as exc:
        log.warning("backtranslation_error", lang=pivot_lang, error=str(exc))
        return None


async def _backtranslate_local(text: str, pivot_lang: str) -> str | None:
    """Local back-translation using Helsinki-NLP/opus-mt models."""
    try:
        from transformers import pipeline  # type: ignore
        import torch

        device = 0 if torch.cuda.is_available() else -1

        forward = pipeline(
            "translation",
            model=f"Helsinki-NLP/opus-mt-en-{pivot_lang}",
            device=device,
        )
        backward = pipeline(
            "translation",
            model=f"Helsinki-NLP/opus-mt-{pivot_lang}-en",
            device=device,
        )

        intermediate = forward(text, max_length=512)[0]["translation_text"]
        result = backward(intermediate, max_length=512)[0]["translation_text"]
        return result

    except Exception as exc:
        log.warning("local_backtranslation_error", error=str(exc))
        return None


async def _grammar_correct(text: str) -> str | None:
    """
    Apply grammar correction using T5-based grammar correction model.
    Alters token-level patterns slightly, enough to fool perplexity detectors.
    """
    try:
        from transformers import pipeline  # type: ignore
        import torch

        corrector = pipeline(
            "text2text-generation",
            model="grammarly/coedit-large",
            device=0 if torch.cuda.is_available() else -1,
        )
        result = corrector(
            f"Fix the grammar: {text}",
            max_length=512,
            num_beams=5,
        )
        return result[0]["generated_text"]
    except Exception as exc:
        log.warning("grammar_correction_error", error=str(exc))
        return None


def _mix_human_ai(ai_text: str, human_text: str) -> str:
    """
    Interleave sentences from AI and human text.
    Creates hard mixed samples that test boundary detection.
    """
    ai_sents = [s.strip() for s in ai_text.split(".") if len(s.strip()) > 20]
    hu_sents = [s.strip() for s in human_text.split(".") if len(s.strip()) > 20]

    if not ai_sents or not hu_sents:
        return ai_text

    # Randomly replace ~30-50% of AI sentences with human sentences
    replace_ratio = random.uniform(0.3, 0.5)
    n_replace = max(1, int(len(ai_sents) * replace_ratio))
    replace_indices = random.sample(range(len(ai_sents)), min(n_replace, len(ai_sents)))

    mixed = list(ai_sents)
    for idx in replace_indices:
        if hu_sents:
            mixed[idx] = random.choice(hu_sents)

    return ". ".join(mixed) + "."


# ── Main pipeline ─────────────────────────────────────────────

async def generate_adversarial_samples(
    ai_generated_path: Path = Path("datasets/ai-generated/ai_generated.jsonl"),
    human_path: Path = Path("datasets/human/human.jsonl"),
    output_dir: Path = OUTPUT_DIR,
    adversarial_ratio: float = 0.15,  # 15% of AI set becomes adversarial
    concurrency: int = 3,
) -> Path:
    """
    Read AI-generated samples, apply evasion transforms, write adversarial set.

    Attack distribution:
      40% paraphrase  — most common real-world evasion
      30% back-translation — moderate evasion
      20% grammar correction — subtle evasion
      10% mixed human/AI — hardest to detect

    Args:
        ai_generated_path: Source AI-generated JSONL.
        human_path:        Source human JSONL (for mixing).
        output_dir:        Output directory.
        adversarial_ratio: Fraction of AI samples to attack.
        concurrency:       Max concurrent transform calls.

    Returns:
        Path to adversarial.jsonl
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "adversarial.jsonl"

    # Load AI samples
    ai_records: list[dict] = []
    if ai_generated_path.exists():
        with ai_generated_path.open() as f:
            for line in f:
                try:
                    ai_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Load human samples for mixing
    human_texts: list[str] = []
    if human_path.exists():
        with human_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    human_texts.append(rec["text"])
                except (json.JSONDecodeError, KeyError):
                    continue

    n_to_attack = int(len(ai_records) * adversarial_ratio)
    attack_pool = random.sample(ai_records, min(n_to_attack, len(ai_records)))

    log.info(
        "generating_adversarial",
        total_ai=len(ai_records),
        to_attack=len(attack_pool),
    )

    semaphore = asyncio.Semaphore(concurrency)
    written = 0

    # Attack type probabilities
    attack_weights = [
        ("paraphrase", 0.40),
        ("backtranslation", 0.30),
        ("grammar_correction", 0.20),
        ("mixed", 0.10),
    ]
    attack_types, attack_probs = zip(*attack_weights)

    async def _attack_one(record: dict) -> dict | None:
        async with semaphore:
            text = record["text"]
            attack: AttackType = random.choices(attack_types, weights=attack_probs, k=1)[0]  # type: ignore

            result_text: str | None = None

            if attack == "paraphrase":
                result_text = await _paraphrase(text)
            elif attack == "backtranslation":
                lang = random.choice(PIVOT_LANGUAGES)
                result_text = await _backtranslate(text, lang)
                attack = f"backtranslation_{lang}"  # type: ignore
            elif attack == "grammar_correction":
                result_text = await _grammar_correct(text)
            elif attack == "mixed" and human_texts:
                human = random.choice(human_texts)
                result_text = _mix_human_ai(text, human)

            if not result_text or len(result_text.split()) < 20:
                return None

            return {
                "text": result_text.strip(),
                "label": 1,                     # still AI-origin
                "original_model": record.get("model", "unknown"),
                "attack_type": attack,
                "original_text": text[:200],     # keep snippet for debugging
                "word_count": len(result_text.split()),
            }

    tasks = [asyncio.create_task(_attack_one(r)) for r in attack_pool]

    with out_path.open("w", encoding="utf-8") as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
                if written % 200 == 0:
                    log.info("adversarial_progress", written=written, total=len(attack_pool))

    log.info("adversarial_complete", written=written, path=str(out_path))
    return out_path


if __name__ == "__main__":
    asyncio.run(generate_adversarial_samples())
