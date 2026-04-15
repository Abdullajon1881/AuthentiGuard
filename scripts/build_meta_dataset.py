"""
Stage 2 — Part 1: build the training dataset for the stacking meta-classifier.

Loads `datasets/processed/val.parquet`, runs the production `TextDetector`
on every row (same loader shape as `backend/app/workers/text_worker.py::
_get_detector`: adversarial_checkpoint=None, meta_checkpoint=None,
device="cpu"), and writes a parquet with per-layer scores, the current
fixed-weight ensemble score, and the true label.

Output columns (5):
    l1_score            -- perplexity layer (GPT-2)
    l2_score            -- stylometry layer (spaCy)
    l3_score            -- semantic layer (DeBERTa-v3-small)
    final_score_current -- what the current fixed-weight combiner produces
    label               -- 0 = human, 1 = AI

Parquet file-level metadata carries: git_sha, timestamp_utc, source_sha256,
n_samples, checkpoint_path. These are also mirrored in a sidecar JSON
alongside the parquet for easy diffing.

Deterministic. Read-only against the val split. Writes only the output
parquet and the sidecar JSON. Does NOT touch any test split.

Usage:
    python scripts/build_meta_dataset.py \
        --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
        --data-dir datasets/processed \
        --split val \
        --output datasets/meta/meta_train.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_production_detector(checkpoint: Path, device: str) -> Any:
    """Construct TextDetector with the EXACT production arguments.

    Mirrors backend/app/workers/text_worker.py::_get_detector. Keeps
    adversarial_checkpoint=None and meta_checkpoint=None so the layer
    scores we observe match exactly what production sees.
    """
    from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore
    detector = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device=device,
    )
    detector.load_models()
    return detector


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", default="val", help="parquet split basename (default: val)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    split_path = args.data_dir / f"{args.split}.parquet"
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    # This script is explicitly restricted to val — the spec says
    # "Load validation split", and we do not touch test.
    if args.split not in ("val", "validation"):
        print(
            f"WARNING: --split={args.split!r} is not a validation split. "
            "The meta dataset should be built from val to keep test honest.",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_meta] loading dataset: {split_path}")
    import pyarrow.parquet as pq
    table = pq.read_table(split_path)
    if "text" not in table.column_names or "label" not in table.column_names:
        print(
            f"ERROR: {split_path} must have columns `text` and `label`. "
            f"Got: {table.column_names}",
            file=sys.stderr,
        )
        return 2
    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[build_meta] rows: {len(rows)}")

    print(f"[build_meta] loading detector from: {args.checkpoint}")
    t_load_start = time.time()
    detector = _build_production_detector(args.checkpoint, args.device)
    t_load_end = time.time()
    print(f"[build_meta] detector loaded in {t_load_end - t_load_start:.1f}s")
    print(f"[build_meta] active layers: {detector._active_layers}")

    l1_scores: list[float] = []
    l2_scores: list[float] = []
    l3_scores: list[float] = []
    final_scores: list[float] = []
    labels: list[int] = []

    use_tqdm = not args.no_progress
    try:
        if use_tqdm:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(rows, desc="build_meta", unit="row")
        else:
            iterator = rows
    except ImportError:
        iterator = rows

    t_analyze_start = time.time()
    for row in iterator:
        text = row["text"]
        true_label = int(row["label"])

        # Run the full production pipeline so we capture exactly what
        # the detector outputs. `result.layer_results` preserves the
        # order L1, L2, L3 on the 3-layer production path.
        result = detector.analyze(text)
        by_name = {r.layer_name: r for r in result.layer_results}

        def _s(name: str) -> float:
            r = by_name.get(name)
            # Neutral 0.5 on an errored layer — matches how the production
            # fallback combiner already treats layer errors.
            return float(r.score) if (r is not None and not r.error) else 0.5

        l1_scores.append(_s("perplexity"))
        l2_scores.append(_s("stylometry"))
        l3_scores.append(_s("transformer"))
        final_scores.append(float(result.score))
        labels.append(true_label)

    t_analyze_end = time.time()
    print(f"[build_meta] analyze complete in {t_analyze_end - t_analyze_start:.1f}s")

    # Build and write parquet with file-level metadata
    import pyarrow as pa
    out_table = pa.table(
        {
            "l1_score": pa.array(l1_scores, type=pa.float64()),
            "l2_score": pa.array(l2_scores, type=pa.float64()),
            "l3_score": pa.array(l3_scores, type=pa.float64()),
            "final_score_current": pa.array(final_scores, type=pa.float64()),
            "label": pa.array(labels, type=pa.int32()),
        }
    )

    git_sha = _git_sha()
    source_sha = _sha256_file(split_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        b"git_sha": git_sha.encode("ascii"),
        b"timestamp_utc": timestamp.encode("ascii"),
        b"source_split": str(split_path).replace("\\", "/").encode("utf-8"),
        b"source_sha256": source_sha.encode("ascii"),
        b"checkpoint": str(args.checkpoint).replace("\\", "/").encode("utf-8"),
        b"n_samples": str(len(rows)).encode("ascii"),
        b"active_layers": str(list(detector._active_layers)).encode("ascii"),
    }
    out_table = out_table.replace_schema_metadata(metadata)

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(out_table, tmp_path)
    os.replace(tmp_path, args.output)
    print(f"[build_meta] wrote {args.output} ({len(rows)} rows)")

    # Sidecar JSON with the same metadata — easier to diff than parquet
    # schema metadata and mirrors the artifact style of Stage 1.
    sidecar_path = args.output.with_suffix(".meta.json")
    sidecar = {
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "source_split": str(split_path).replace("\\", "/"),
        "source_sha256": source_sha,
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "n_samples": len(rows),
        "active_layers": list(detector._active_layers),
        "columns": ["l1_score", "l2_score", "l3_score", "final_score_current", "label"],
        "label_distribution": {
            "0": int(sum(1 for y in labels if y == 0)),
            "1": int(sum(1 for y in labels if y == 1)),
        },
        "wallclock_seconds": {
            "load": round(t_load_end - t_load_start, 2),
            "analyze": round(t_analyze_end - t_analyze_start, 2),
            "total": round(time.time() - t_load_start, 2),
        },
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[build_meta] wrote {sidecar_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
