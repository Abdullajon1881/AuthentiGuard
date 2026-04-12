"""
Phase 1 validation: end-to-end test of calibrated L1+L2 text detection pipeline.

Runs 10 human + 10 AI samples through the real TextDetector and prints a
results table with score, label, confidence, layer scores, and timing.

Two modes:
  --mode direct   (default) Instantiate TextDetector in-process. No Docker needed.
  --mode api      Hit the running API at --base-url. Requires Docker stack up.

Usage:
    python scripts/test_text_detector.py
    python scripts/test_text_detector.py --mode api --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Test samples ──────────────────────────────────────────────

SAMPLES = [
    # ── HUMAN (10) ──────────────────────────────────────────
    {
        "text": "The fire started around 2 a.m. in a warehouse on the city's east side, according to officials. Three engine companies responded and had the blaze under control by dawn. No injuries were reported, but the building sustained significant structural damage. Investigators haven't determined a cause yet.",
        "label": "human",
        "type": "news",
    },
    {
        "text": "The Congo River is the deepest river in the world, with measured depths exceeding 220 metres. It drains an area of nearly 4 million square kilometres across central Africa. Unlike most major rivers, it crosses the equator twice. Its lower course features a series of cataracts that prevent ocean-going ships from reaching the interior.",
        "label": "human",
        "type": "wikipedia",
    },
    {
        "text": "I've been trying to get into sourdough baking for about three months now and honestly it's been a rollercoaster. My first loaf was basically a brick. Like, I could've used it as a doorstop. But last week's attempt actually had some decent oven spring, and the crumb was open enough that I didn't feel embarrassed showing it to my wife.",
        "label": "human",
        "type": "blog_casual",
    },
    {
        "text": "lmao my roommate just tried to make scrambled eggs and somehow set off every smoke alarm in the building. like HOW. it's eggs. the fire department actually showed up and he had to explain to two firefighters that he was just making breakfast. I can't live with this man anymore",
        "label": "human",
        "type": "reddit_casual",
    },
    {
        "text": "TCP uses a three-way handshake to establish connections. The client sends a SYN packet, the server responds with SYN-ACK, and the client completes the handshake with an ACK. This process ensures both sides agree on initial sequence numbers before data transfer begins. If any step fails, the connection isn't established and the client typically retries after a timeout.",
        "label": "human",
        "type": "technical",
    },
    {
        "text": "My daughter asked me why the sky is blue and I realized I couldn't actually explain it properly. Something about light scattering? I remember learning it in school but the details are gone. Ended up just saying 'that's how the atmosphere works' which she did NOT accept. She's six and apparently already a better scientist than me.",
        "label": "human",
        "type": "personal_story",
    },
    {
        "text": "Okay so I finally read Dune and I get it now. I understand the hype. But can we talk about how the pacing in the second half is completely different from the first? It goes from this slow political thriller to Paul basically speedrunning a hero's journey in like 80 pages. Still loved it but that shift caught me off guard.",
        "label": "human",
        "type": "book_review",
    },
    {
        "text": "The election results in the northern districts were closer than anyone predicted. The incumbent won by just 340 votes out of nearly 50,000 cast, triggering an automatic recount. Both campaigns have sent legal teams. Local officials say the recount could take up to two weeks given the volume of mail-in ballots that need to be re-examined.",
        "label": "human",
        "type": "news_politics",
    },
    {
        "text": "Been woodworking for about 15 years and I still mess up dovetails. You'd think by now I'd have it down but nope, every third joint has a gap somewhere. My buddy who's been doing it half as long makes perfect ones every time. Some people just have steadier hands I guess. Anyway the bookshelf turned out fine, you can't see the mistakes from across the room.",
        "label": "human",
        "type": "hobby_forum",
    },
    {
        "text": "The restaurant on the corner finally reopened after the renovation and it's... fine? The food is basically the same, which is good, but they replaced all the booths with these tiny two-tops and the vibe is totally different now. It used to feel like a diner and now it feels like a fast casual place trying to be upscale. The burgers are still great though.",
        "label": "human",
        "type": "local_review",
    },

    # ── AI (10) ─────────────────────────────────────────────
    {
        "text": "Artificial intelligence is fundamentally transforming the landscape of modern healthcare. Machine learning algorithms are enabling earlier and more accurate disease detection, while natural language processing systems are streamlining clinical documentation. These technological advancements hold tremendous promise for improving patient outcomes, reducing healthcare costs, and addressing persistent inequities in access to quality medical care.",
        "label": "ai",
        "type": "chatgpt_essay",
    },
    {
        "text": "The concept of sustainable development encompasses a holistic approach to balancing economic growth, environmental stewardship, and social equity. Organizations worldwide are increasingly recognizing that long-term prosperity requires careful consideration of ecological impacts and community well-being. By integrating sustainability principles into core business strategies, companies can create value for shareholders while simultaneously contributing to broader societal goals.",
        "label": "ai",
        "type": "chatgpt_corporate",
    },
    {
        "text": "Understanding the nuances of effective project management requires a comprehensive examination of several interconnected factors. Successful project delivery depends on clear communication channels, well-defined scope parameters, and robust risk mitigation strategies. Furthermore, fostering a collaborative team environment and maintaining stakeholder alignment throughout the project lifecycle are essential components of achieving desired outcomes within established timelines and budget constraints.",
        "label": "ai",
        "type": "claude_professional",
    },
    {
        "text": "The evolution of programming languages reflects the broader trajectory of computer science as a discipline. From early machine code and assembly languages to modern high-level frameworks, each generation has introduced abstractions that enable developers to express increasingly complex ideas with greater clarity and efficiency. This progression has democratized software development, making it accessible to a wider range of practitioners.",
        "label": "ai",
        "type": "ai_technical",
    },
    {
        "text": "Effective leadership in the modern workplace requires a multifaceted approach that balances strategic vision with emotional intelligence. Leaders must navigate increasingly complex organizational dynamics while fostering innovation and maintaining team cohesion. Research consistently demonstrates that leaders who prioritize transparent communication, inclusive decision-making, and continuous professional development create more resilient and productive organizations.",
        "label": "ai",
        "type": "ai_leadership",
    },
    {
        "text": "The Mediterranean diet has garnered significant attention from researchers and health professionals due to its well-documented health benefits. Characterized by an emphasis on fruits, vegetables, whole grains, legumes, and olive oil, this dietary pattern has been associated with reduced cardiovascular risk, improved cognitive function, and enhanced longevity. Additionally, the Mediterranean diet promotes sustainable food systems that align with environmental conservation goals.",
        "label": "ai",
        "type": "ai_health",
    },
    {
        "text": "Digital transformation represents a fundamental shift in how organizations leverage technology to create value. This process extends beyond merely adopting new tools; it requires reimagining business processes, organizational culture, and customer engagement strategies. Companies that successfully navigate digital transformation are better positioned to respond to market disruptions, capitalize on emerging opportunities, and deliver enhanced experiences to their stakeholders.",
        "label": "ai",
        "type": "ai_business",
    },
    {
        "text": "The study of cognitive biases reveals important insights into human decision-making processes. Confirmation bias, anchoring effects, and availability heuristics are among the most prevalent cognitive shortcuts that influence our judgments. Understanding these biases is essential for developing more effective educational approaches, designing better user interfaces, and creating public policies that account for the predictable ways in which human reasoning deviates from purely rational models.",
        "label": "ai",
        "type": "ai_psychology",
    },
    {
        "text": "Renewable energy infrastructure development requires careful consideration of multiple interdependent factors. Site selection, grid integration, environmental impact assessment, and community engagement all play critical roles in determining project viability and success. Moreover, advances in energy storage technology are addressing the intermittency challenges traditionally associated with solar and wind power, making renewable sources increasingly competitive with conventional fossil fuel generation.",
        "label": "ai",
        "type": "ai_energy",
    },
    {
        "text": "The intersection of data science and public health offers unprecedented opportunities for improving population-level health outcomes. Advanced analytics techniques enable researchers to identify disease patterns, predict outbreak trajectories, and evaluate intervention effectiveness with greater precision than ever before. However, realizing the full potential of data-driven public health requires addressing challenges related to data quality, privacy protections, and equitable access to technological resources.",
        "label": "ai",
        "type": "ai_datascience",
    },
]


def run_direct() -> list[dict]:
    """Run TextDetector in-process (no API, no Docker)."""
    from ai.text_detector.ensemble.text_detector import TextDetector

    print("Loading TextDetector (L1+L2 MVP mode)...")
    print("  GPT-2 will download on first run (~500MB, 5-10 min)")
    t0 = time.time()

    detector = TextDetector(
        transformer_checkpoint=None,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
    )
    detector.load_models()
    load_time = time.time() - t0
    print(f"  Models loaded in {load_time:.1f}s\n")

    results = []
    for i, sample in enumerate(SAMPLES):
        t_start = time.time()
        result = detector.analyze(sample["text"])
        elapsed_ms = int((time.time() - t_start) * 1000)

        layer_scores = {r.layer_name: r.score for r in result.layer_results}

        results.append({
            "type": sample["type"],
            "true_label": sample["label"],
            "score": result.score,
            "label": result.label,
            "confidence": result.confidence,
            "processing_ms": elapsed_ms,
            "layer_scores": layer_scores,
            "n_sentences": len(result.evidence_summary.get("sentence_scores", [])),
            "top_signals": result.evidence_summary.get("top_signals", []),
        })

        print(f"  [{i+1}/{len(SAMPLES)}] {sample['type']:<20} "
              f"score={result.score:.3f} label={result.label:<11} "
              f"{elapsed_ms}ms")

    return results


def run_api(base_url: str) -> list[dict]:
    """Submit samples via the API and poll for results."""
    import urllib.request
    import urllib.error

    results = []
    for i, sample in enumerate(SAMPLES):
        # Submit
        req_data = json.dumps({
            "text": sample["text"],
            "content_type": "text",
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/api/v1/analyze/text",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                submit_resp = json.loads(resp.read())
        except urllib.error.URLError as e:
            print(f"  ERROR submitting sample {i+1}: {e}")
            continue

        job_id = submit_resp["job_id"]

        # Poll
        for attempt in range(60):
            time.sleep(2)
            status_req = urllib.request.Request(
                f"{base_url}/api/v1/jobs/{job_id}"
            )
            with urllib.request.urlopen(status_req) as resp:
                status_resp = json.loads(resp.read())
            if status_resp["status"] == "completed":
                break
            if status_resp["status"] == "failed":
                print(f"  [{i+1}] FAILED: {sample['type']}")
                break
        else:
            print(f"  [{i+1}] TIMEOUT: {sample['type']}")
            continue

        # Fetch result
        result_req = urllib.request.Request(
            f"{base_url}/api/v1/jobs/{job_id}/result"
        )
        with urllib.request.urlopen(result_req) as resp:
            result_resp = json.loads(resp.read())

        ls = result_resp.get("layer_scores", {})
        results.append({
            "type": sample["type"],
            "true_label": sample["label"],
            "score": result_resp["authenticity_score"],
            "label": result_resp["label"],
            "confidence": result_resp["confidence"],
            "processing_ms": result_resp.get("processing_ms"),
            "layer_scores": ls,
            "n_sentences": len(result_resp.get("sentence_scores", [])),
            "top_signals": result_resp.get("top_signals", []),
        })

        print(f"  [{i+1}/{len(SAMPLES)}] {sample['type']:<20} "
              f"score={result_resp['authenticity_score']:.3f} "
              f"label={result_resp['label']:<11} "
              f"{result_resp.get('processing_ms', '?')}ms")

    return results


def print_results(results: list[dict]) -> None:
    """Print formatted results table and anomaly flags."""

    # Header
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)

    print(f"\n{'TYPE':<22} {'TRUE':>5} {'SCORE':>6} {'LABEL':<11} "
          f"{'CONF':>5} {'MS':>6} {'PPL':>5} {'STY':>5} {'TFM':>5} {'ADV':>5}")
    print("-" * 100)

    correct = 0
    anomalies = []

    for r in results:
        ls = r["layer_scores"]
        ppl = ls.get("perplexity", "—")
        sty = ls.get("stylometry", "—")
        tfm = ls.get("transformer", "—")
        adv = ls.get("adversarial", "—")

        ppl_str = f"{ppl:.3f}" if isinstance(ppl, (int, float)) else ppl
        sty_str = f"{sty:.3f}" if isinstance(sty, (int, float)) else sty
        tfm_str = f"{tfm:.3f}" if isinstance(tfm, (int, float)) else tfm
        adv_str = f"{adv:.3f}" if isinstance(adv, (int, float)) else adv

        # Determine correctness
        pred_is_ai = r["label"] == "AI"
        true_is_ai = r["true_label"] == "ai"
        is_correct = (pred_is_ai == true_is_ai) or r["label"] == "UNCERTAIN"
        if pred_is_ai == true_is_ai:
            correct += 1
        marker = " " if is_correct else " <<"

        print(f"{r['type']:<22} {r['true_label']:>5} {r['score']:>6.3f} "
              f"{r['label']:<11} {r['confidence']:>5.3f} "
              f"{r['processing_ms'] or 0:>6} "
              f"{ppl_str:>5} {sty_str:>5} {tfm_str:>5} {adv_str:>5}{marker}")

        # Flag anomalies
        flags = []
        if r["confidence"] < 0.55:
            flags.append(f"low confidence ({r['confidence']:.3f})")
        if r["true_label"] == "ai" and r["score"] < 0.55:
            flags.append(f"AI text scored low ({r['score']:.3f})")
        if r["true_label"] == "human" and r["score"] > 0.60:
            flags.append(f"human text scored high ({r['score']:.3f})")
        if flags:
            anomalies.append((r["type"], flags))

    # Summary
    total = len(results)
    human_results = [r for r in results if r["true_label"] == "human"]
    ai_results = [r for r in results if r["true_label"] == "ai"]

    human_correct = sum(1 for r in human_results if r["label"] == "HUMAN")
    ai_correct = sum(1 for r in ai_results if r["label"] == "AI")
    human_uncertain = sum(1 for r in human_results if r["label"] == "UNCERTAIN")
    ai_uncertain = sum(1 for r in ai_results if r["label"] == "UNCERTAIN")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"  Total samples:      {total}")
    print(f"  Correct:            {correct}/{total} ({correct/total:.0%})")
    print(f"  Human correct:      {human_correct}/{len(human_results)} "
          f"(+{human_uncertain} uncertain)")
    print(f"  AI correct:         {ai_correct}/{len(ai_results)} "
          f"(+{ai_uncertain} uncertain)")

    # Score distributions
    h_scores = [r["score"] for r in human_results]
    a_scores = [r["score"] for r in ai_results]
    if h_scores:
        print(f"\n  Human score range:  [{min(h_scores):.3f}, {max(h_scores):.3f}] "
              f"mean={sum(h_scores)/len(h_scores):.3f}")
    if a_scores:
        print(f"  AI score range:     [{min(a_scores):.3f}, {max(a_scores):.3f}] "
              f"mean={sum(a_scores)/len(a_scores):.3f}")

    # Processing time
    times = [r["processing_ms"] for r in results if r["processing_ms"]]
    if times:
        print(f"\n  Avg processing:     {sum(times)/len(times):.0f}ms")
        print(f"  Max processing:     {max(times)}ms")

    # Layer availability
    sample_ls = results[0]["layer_scores"] if results else {}
    active = [k for k, v in sample_ls.items() if v is not None]
    inactive = [k for k, v in sample_ls.items() if v is None]
    print(f"\n  Active layers:      {', '.join(active) if active else 'none'}")
    if inactive:
        print(f"  Inactive layers:    {', '.join(inactive)} (no checkpoint)")

    # Anomalies
    if anomalies:
        print(f"\n  {'='*60}")
        print("  ANOMALIES")
        print(f"  {'='*60}")
        for sample_type, flags in anomalies:
            for flag in flags:
                print(f"  [{sample_type}] {flag}")
    else:
        print("\n  No anomalies detected.")

    # Schema check
    print(f"\n  Schema validation:")
    schema_ok = True
    for r in results:
        if not (0.0 <= r["score"] <= 1.0):
            print(f"    FAIL: score out of range: {r['score']}")
            schema_ok = False
        if not (0.0 <= r["confidence"] <= 1.0):
            print(f"    FAIL: confidence out of range: {r['confidence']}")
            schema_ok = False
        if r["label"] not in ("AI", "HUMAN", "UNCERTAIN"):
            print(f"    FAIL: invalid label: {r['label']}")
            schema_ok = False
    if schema_ok:
        print("    All scores in [0,1], labels valid, schema OK")


def main():
    parser = argparse.ArgumentParser(description="Validate text detector pipeline")
    parser.add_argument("--mode", choices=["direct", "api"], default="direct",
                        help="direct = in-process, api = via HTTP")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="API base URL (api mode only)")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Samples: {len(SAMPLES)} ({sum(1 for s in SAMPLES if s['label']=='human')} human, "
          f"{sum(1 for s in SAMPLES if s['label']=='ai')} AI)\n")

    t0 = time.time()
    if args.mode == "direct":
        results = run_direct()
    else:
        results = run_api(args.base_url)

    total_time = time.time() - t0
    print_results(results)
    print(f"\n  Total wall time:    {total_time:.1f}s")


if __name__ == "__main__":
    main()
