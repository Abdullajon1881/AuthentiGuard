"""
Dataset configuration for AuthentiGuard text detection pipeline.
All source definitions, generation parameters, and split targets live here.
"""

from dataclasses import dataclass, field
from typing import Literal

# ── Targets ────────────────────────────────────────────────────
INITIAL_TARGET = 100_000       # Phase 1 target
SCALE_TARGET   = 1_000_000    # Phase 2 scale target

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# 50 / 50 human vs AI per roadmap
HUMAN_RATIO = 0.50
AI_RATIO    = 0.50


# ── Human sources ─────────────────────────────────────────────
@dataclass
class HumanSource:
    name: str
    hf_dataset: str          # HuggingFace dataset path
    hf_config: str | None
    hf_split: str
    text_column: str
    min_words: int = 50
    max_words: int = 1000
    target_samples: int = 0  # filled at runtime


HUMAN_SOURCES: list[HumanSource] = [
    HumanSource(
        name="wikipedia",
        hf_dataset="wikipedia",
        hf_config="20220301.en",
        hf_split="train",
        text_column="text",
        min_words=100,
        max_words=800,
    ),
    HumanSource(
        name="reddit",
        hf_dataset="webis/tldr-17",
        hf_config=None,
        hf_split="train",
        text_column="content",
        min_words=50,
        max_words=500,
    ),
    HumanSource(
        name="arxiv",
        hf_dataset="scientific_papers",
        hf_config="arxiv",
        hf_split="train",
        text_column="abstract",
        min_words=80,
        max_words=400,
    ),
    HumanSource(
        name="news",
        hf_dataset="cc_news",
        hf_config=None,
        hf_split="train",
        text_column="text",
        min_words=100,
        max_words=800,
    ),
    HumanSource(
        name="books",
        hf_dataset="bookcorpus",
        hf_config=None,
        hf_split="train",
        text_column="text",
        min_words=100,
        max_words=600,
    ),
]


# ── AI generation parameters ──────────────────────────────────
@dataclass
class GenerationConfig:
    model: str
    provider: Literal["openai", "anthropic", "together", "local"]
    temperatures: list[float]
    system_prompts: list[str]
    user_prompts_template: str   # {topic} placeholder
    max_tokens: int = 600
    weight: float = 1.0          # sampling weight across models


AI_GENERATION_CONFIGS: list[GenerationConfig] = [
    GenerationConfig(
        model="gpt-4-turbo",
        provider="openai",
        temperatures=[0.3, 0.7, 1.0, 1.2],
        system_prompts=[
            "You are a knowledgeable writer. Write clearly and informatively.",
            "Write in a casual, conversational style.",
            "Write in a formal, academic style.",
            "Write as if for a general audience blog post.",
        ],
        user_prompts_template="Write a {length}-word passage about {topic}.",
        max_tokens=800,
        weight=0.30,
    ),
    GenerationConfig(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        temperatures=[0.3, 0.7, 1.0],
        system_prompts=[
            "You are a helpful writing assistant.",
            "Write in a thoughtful, nuanced style.",
            "Write in a direct, journalistic style.",
        ],
        user_prompts_template="Write a {length}-word passage about {topic}.",
        max_tokens=800,
        weight=0.25,
    ),
    GenerationConfig(
        model="meta-llama/Llama-3-70b-chat-hf",
        provider="together",
        temperatures=[0.5, 0.8, 1.1],
        system_prompts=[
            "You are a writer. Produce clear, informative text.",
            "Write naturally, as a human would.",
        ],
        user_prompts_template="Write a {length}-word passage about {topic}.",
        max_tokens=800,
        weight=0.25,
    ),
    GenerationConfig(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        provider="together",
        temperatures=[0.5, 0.9, 1.2],
        system_prompts=[
            "You are a knowledgeable assistant. Write informatively.",
            "Write in a conversational but informative style.",
        ],
        user_prompts_template="Write a {length}-word passage about {topic}.",
        max_tokens=800,
        weight=0.20,
    ),
]


# ── Topics for AI generation ──────────────────────────────────
GENERATION_TOPICS: list[str] = [
    # Science & tech
    "climate change and renewable energy",
    "artificial intelligence and machine learning",
    "quantum computing fundamentals",
    "CRISPR gene editing",
    "the history of the internet",
    "space exploration and Mars missions",
    "nuclear fusion energy",
    "blockchain technology",
    # Society & culture
    "the impact of social media on mental health",
    "remote work and productivity",
    "urban planning and smart cities",
    "the gig economy",
    "diversity and inclusion in the workplace",
    "the future of education",
    "healthcare systems worldwide",
    "income inequality",
    # History & politics
    "the causes of World War I",
    "the Cold War and its legacy",
    "the rise of democracy in the 20th century",
    "colonialism and its effects",
    "the United Nations and global governance",
    # Nature & environment
    "biodiversity and ecosystem collapse",
    "ocean acidification",
    "deforestation and reforestation",
    "water scarcity",
    # Arts & philosophy
    "the philosophy of consciousness",
    "ethics of artificial intelligence",
    "modern art movements",
    "the psychology of creativity",
    "stoicism and modern life",
]

GENERATION_LENGTHS = [100, 150, 200, 300, 400, 500]  # word counts
