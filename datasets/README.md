# AuthentiGuard Datasets

All datasets are version-controlled with DVC. Raw files are stored in S3/MinIO,
not in git. Only `.dvc` pointer files are committed.

## Directory Structure

```
datasets/
├── human/
│   └── human.jsonl           ← Wikipedia, Reddit, arXiv, news, books
├── ai-generated/
│   └── ai_generated.jsonl    ← GPT-4, Claude, LLaMA, Mistral
├── adversarial/
│   └── adversarial.jsonl     ← Paraphrase, backtranslation, grammar correction, mixed
└── processed/
    ├── train.parquet          ← 80% of total
    ├── val.parquet            ← 10% of total
    ├── test.parquet           ← 10% of total
    └── dataset_card.json      ← Stats and metadata
```

## Targets

| Phase | Total Samples | Human | AI-Generated | Adversarial |
|-------|--------------|-------|--------------|-------------|
| Phase 1 (initial) | 100K | 50K | 50K | ~7.5K (15% of AI) |
| Phase 2 (scale)   | 1M+  | 500K | 500K | ~75K |

## Human Sources

| Source | HuggingFace Dataset | Text Type |
|--------|-------------------|-----------|
| Wikipedia | `wikipedia/20220301.en` | Encyclopedic articles |
| Reddit | `webis/tldr-17` | Social posts |
| arXiv | `scientific_papers/arxiv` | Academic abstracts |
| News | `cc_news` | News articles |
| Books | `bookcorpus` | Book passages |

## AI Generation

| Model | Provider | Weight |
|-------|----------|--------|
| GPT-4 Turbo | OpenAI | 30% |
| Claude 3.5 Sonnet | Anthropic | 25% |
| LLaMA 3 70B | Together AI | 25% |
| Mixtral 8x7B | Together AI | 20% |

Each model is run at multiple temperatures (0.3, 0.7, 1.0, 1.2) and across
multiple system prompts to maximize diversity.

## Adversarial Attacks

| Attack | Fraction | Description |
|--------|---------|-------------|
| Paraphrase | 40% | GPT-4 full rewrite |
| Back-translation | 30% | EN→pivot→EN via 8 languages |
| Grammar correction | 20% | T5-based grammar correction |
| Mixed human/AI | 10% | Interleaved sentences |

## Running the Pipeline

```bash
# Full pipeline (first run):
python -m ai.text_detector.data_pipeline.run_pipeline --all

# Or with DVC (tracks outputs automatically):
dvc repro

# Scale to 1M samples:
python -m ai.text_detector.data_pipeline.run_pipeline --all --target 1000000

# Individual stages:
python -m ai.text_detector.data_pipeline.run_pipeline --human
python -m ai.text_detector.data_pipeline.run_pipeline --ai
python -m ai.text_detector.data_pipeline.run_pipeline --adversarial
python -m ai.text_detector.data_pipeline.run_pipeline --preprocess
```

## DVC Usage

```bash
# Push dataset to remote storage:
dvc push

# Pull dataset on a new machine:
dvc pull

# Check what would be run:
dvc status
```

## Record Schema

Every JSONL record has this structure:

```json
{
  "text":       "...",     // cleaned text content
  "label":      0,         // 0 = human, 1 = AI-generated
  "source":     "...",     // human: source name; AI: model name
  "word_count": 150,       // post-cleaning word count
  "sentences":  ["..."],   // sentence-tokenized (added in preprocessing)
  "n_sentences": 8         // sentence count
}
```

AI records additionally include: `model`, `provider`, `temperature`, `topic`, `system_prompt`.
Adversarial records additionally include: `attack_type`, `original_model`, `original_text` (snippet).
