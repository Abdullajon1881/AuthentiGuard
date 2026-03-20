"""
Step 82: Model attribution classifier.

Identifies which AI model family generated the content.
Output: percentage breakdown across model families.

Model families:
  GPT-family    — OpenAI GPT-3.5, GPT-4, GPT-4o, o1, o3
  Claude-family — Anthropic Claude 1/2/3 (Haiku, Sonnet, Opus)
  LLaMA-family  — Meta LLaMA 2/3, Mistral, Mixtral, Vicuna, Alpaca
  Human         — Human-written content
  Other-AI      — DALL-E, Midjourney, Stable Diffusion, Gemini, etc.

Approach for text:
  Train a multi-class classifier on token probability curves, sentence
  transition patterns, and stylometric features that are specific to
  each model family. Different models have different "verbal habits":

  GPT-4:   Prefers "Furthermore", "Moreover", em dashes, long nested sentences
  Claude:  Prefers "I'd be happy to", structured bullet reasoning, disclaimers
  LLaMA:   More repetitive, shorter sentences, fewer subordinate clauses
  Human:   Variable, contains errors, personal voice, idiosyncratic choices

For images/video:
  GAN/diffusion models leave different spectral fingerprints.
  StyleGAN → characteristic high-frequency grid pattern
  Stable Diffusion → characteristic mid-frequency smoothing pattern
  DALL-E → characteristic colour saturation distribution

This module provides both a trained classifier (post Phase 10 training)
and a heuristic fallback for immediate use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class AttributionResult:
    """Model attribution breakdown with confidence."""
    gpt_family:    float   # [0,1] — probability content is from GPT-family
    claude_family: float
    llama_family:  float
    human:         float
    other_ai:      float

    primary_attribution:    str     # highest probability class
    primary_confidence:     float   # probability of primary class
    is_confident:           bool    # primary_confidence > 0.5


# ── Text attribution heuristics ───────────────────────────────

# Token/phrase patterns specific to each model family
GPT_MARKERS = {
    "Furthermore,", "Moreover,", "Additionally,", "Consequently,",
    "It's worth noting", "It is worth noting", "In conclusion,",
    "To summarize,", "As mentioned earlier", "This underscores",
    "Delve into", "Tapestry", "Nuanced", "Multifaceted",
    "Robust", "Leverage", "Paradigm", "Facilitate", "Utilize",
}

CLAUDE_MARKERS = {
    "I'd be happy to", "I'd be glad to", "I should mention",
    "I want to be clear", "I should note", "To be clear",
    "Let me", "I'll walk you through", "Here's", "Here are",
    "Happy to help", "Of course!", "Certainly!", "Absolutely!",
    "I notice", "I understand", "thoughtful", "nuanced perspective",
}

LLAMA_MARKERS = {
    "Sure!", "Sure,", "Of course,", "Great question!",
    "I'm just an AI", "I'm a large language model",
    "As a language model", "As an AI assistant",
    "Note that", "Please note", "Important:",
}

HUMAN_MARKERS = {
    "actually", "basically", "you know", "kind of", "sort of",
    "honestly", "tbh", "imo", "idk", "btw", "lol", "omg",
    "anyway", "so yeah", "right?", "I mean", "like,",
    "stuff", "things", "pretty much", "I guess",
}


def attribute_text(
    text: str,
    ai_score: float,
    layer_scores: dict[str, float] | None = None,
) -> AttributionResult:
    """
    Attribute text to a model family using marker analysis + score patterns.

    When a trained classifier is available (Phase 10+), it replaces this
    heuristic implementation but keeps the same output interface.
    """
    if ai_score < 0.40:
        # Clearly human — return human attribution
        return AttributionResult(
            gpt_family=0.02, claude_family=0.02, llama_family=0.01,
            human=0.93, other_ai=0.02,
            primary_attribution="human",
            primary_confidence=0.93,
            is_confident=True,
        )

    words_lower = text.lower()
    n_words     = max(len(text.split()), 1)

    # Count marker occurrences per family
    gpt_count    = sum(1 for m in GPT_MARKERS    if m.lower() in words_lower)
    claude_count = sum(1 for m in CLAUDE_MARKERS  if m.lower() in words_lower)
    llama_count  = sum(1 for m in LLAMA_MARKERS   if m.lower() in words_lower)
    human_count  = sum(1 for m in HUMAN_MARKERS   if m.lower() in words_lower)

    # Normalise by text length
    rate = lambda c: min(c / (n_words / 100), 1.0)
    gpt_rate    = rate(gpt_count)
    claude_rate = rate(claude_count)
    llama_rate  = rate(llama_count)
    human_rate  = rate(human_count)

    # Prior: if AI score is very high, distribute across AI families
    # Base probabilities proportional to marker rates
    total_ai_rate = max(gpt_rate + claude_rate + llama_rate, 0.01)
    human_prob    = (1.0 - ai_score) * (1.0 + human_rate * 0.5)
    ai_pool       = ai_score * (1.0 - human_prob * 0.5)

    gpt_prob    = ai_pool * (gpt_rate    / total_ai_rate) * 0.35 + ai_pool * 0.30
    claude_prob = ai_pool * (claude_rate / total_ai_rate) * 0.35 + ai_pool * 0.20
    llama_prob  = ai_pool * (llama_rate  / total_ai_rate) * 0.30 + ai_pool * 0.20
    other_prob  = ai_pool * 0.10

    # Normalise to sum to 1.0
    total = gpt_prob + claude_prob + llama_prob + human_prob + other_prob
    if total <= 0:
        total = 1.0

    gpt_f    = round(float(gpt_prob    / total), 3)
    claude_f = round(float(claude_prob / total), 3)
    llama_f  = round(float(llama_prob  / total), 3)
    human_f  = round(float(human_prob  / total), 3)
    other_f  = round(max(0.0, 1.0 - gpt_f - claude_f - llama_f - human_f), 3)

    probs = {
        "gpt_family":    gpt_f,
        "claude_family": claude_f,
        "llama_family":  llama_f,
        "human":         human_f,
        "other_ai":      other_f,
    }
    primary = max(probs, key=lambda k: probs[k])
    primary_conf = probs[primary]

    return AttributionResult(
        gpt_family=gpt_f,
        claude_family=claude_f,
        llama_family=llama_f,
        human=human_f,
        other_ai=other_f,
        primary_attribution=primary,
        primary_confidence=primary_conf,
        is_confident=primary_conf > 0.40,
    )


def attribute_image(
    image_score: float,
    fft_features: dict[str, float] | None = None,
    exif_data: dict[str, Any] | None = None,
) -> AttributionResult:
    """
    Attribute an image to a generator family using frequency-domain features.
    """
    fft  = fft_features or {}
    exif = exif_data or {}

    if image_score < 0.40:
        return _human_attribution()

    software = str(exif.get("image_software", "")).lower()
    grid     = fft.get("fft_grid_score", 0.0)
    hf       = fft.get("fft_high_freq_ratio", 0.5)

    # StyleGAN: strong grid artifact
    stylegan_score  = float(grid) * 0.8 + max(0.0, 0.5 - float(hf)) * 0.4
    # Stable Diffusion: medium grid, characteristic colour
    diffusion_score = float(hf) * 0.6 + max(0.0, 0.3 - float(grid)) * 0.4
    # DALL-E: low grid, smooth high-frequency
    dalle_score     = max(0.0, 0.4 - float(grid)) * 0.5 + float(hf) * 0.3

    if "stable diffusion" in software:
        diffusion_score += 0.4
    elif "dall" in software:
        dalle_score += 0.4
    elif "midjourney" in software:
        diffusion_score += 0.3

    ai_pool    = image_score
    total_ai   = max(stylegan_score + diffusion_score + dalle_score, 0.01)
    other_prob = 0.10

    gpt_f    = round(ai_pool * (dalle_score    / total_ai) * (1 - other_prob), 3)
    llama_f  = round(ai_pool * (stylegan_score / total_ai) * (1 - other_prob), 3)
    other_f  = round(ai_pool * other_prob + ai_pool * (diffusion_score / total_ai) * (1 - other_prob), 3)
    human_f  = round(1.0 - image_score, 3)
    claude_f = round(max(0.0, 1.0 - gpt_f - llama_f - other_f - human_f), 3)

    probs = {"gpt_family": gpt_f, "claude_family": claude_f, "llama_family": llama_f,
              "human": human_f, "other_ai": other_f}
    primary = max(probs, key=lambda k: probs[k])

    return AttributionResult(
        gpt_family=gpt_f, claude_family=claude_f,
        llama_family=llama_f, human=human_f, other_ai=other_f,
        primary_attribution=primary,
        primary_confidence=probs[primary],
        is_confident=probs[primary] > 0.40,
    )


def _human_attribution() -> AttributionResult:
    return AttributionResult(
        gpt_family=0.01, claude_family=0.01, llama_family=0.01,
        human=0.96, other_ai=0.01,
        primary_attribution="human",
        primary_confidence=0.96,
        is_confident=True,
    )
