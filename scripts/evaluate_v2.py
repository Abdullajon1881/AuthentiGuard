"""
Evaluate trained model on v2 test split + hard test samples.
Outputs: F1, Precision, Recall, AUROC, confusion matrix.
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report,
)


def load_model(checkpoint_path: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def predict_batch(tokenizer, model, texts: list[str], batch_size: int = 32):
    """Predict labels and probabilities for a batch of texts."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] > 0.5).int()

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_probs)


def evaluate_test_split(tokenizer, model, data_dir: str):
    """Evaluate on v2 test.parquet."""
    import pandas as pd

    test_path = Path(data_dir) / "test.parquet"
    df = pd.read_parquet(test_path)
    texts = df["text"].tolist()
    labels = df["label"].values

    print(f"\n{'='*80}")
    print(f"TEST SPLIT EVALUATION ({len(df)} samples)")
    print(f"{'='*80}")

    preds, probs = predict_batch(tokenizer, model, texts)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    auroc = roc_auc_score(labels, probs)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  AUROC:     {auroc:.4f}")

    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Human  Pred AI")
    print(f"  True Human    {cm[0][0]:<12d}{cm[0][1]}")
    print(f"  True AI       {cm[1][0]:<12d}{cm[1][1]}")

    print(f"\n  Classification Report:")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))

    # Per-source breakdown if source column exists
    if "source" in df.columns:
        print(f"  Per-source accuracy:")
        for src in sorted(df["source"].unique()):
            mask = df["source"] == src
            src_acc = accuracy_score(labels[mask], preds[mask])
            n = mask.sum()
            print(f"    {src:<25s}: {src_acc:.4f} (n={n})")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auroc": auroc}


HARD_SAMPLES = [
    # --- PARAPHRASED AI (humanized) --- should be detected as AI
    {"text": "So basically, yoga is, like, really good for you? It helps with flexibility and stuff, and honestly my anxiety has gotten so much better since I started doing it three months ago. The breathing exercises are kinda weird at first but they actually work lol.", "label": 1, "type": "humanized_AI"},
    {"text": "I tried making sourdough last weekend and it was... okay? Not great. The crust was too thick and the crumb was dense. My friend Sarah said I probably overproofed it but idk. Going to try again this Saturday with a different hydration level.", "label": 1, "type": "humanized_AI"},
    {"text": "ngl the new iPhone is pretty solid. camera upgrade is legit, battery life is better than my old one. only downside is the price tbh. if ur thinking about upgrading from a 14 or older its probably worth it", "label": 1, "type": "humanized_AI_casual"},
    {"text": "The meeting ran way longer than expected. Bob from accounting kept going on about Q3 projections and honestly nobody was paying attention after the first 20 minutes. We need to implement a strict 30-minute cap on these things.", "label": 1, "type": "humanized_AI"},
    {"text": "My cat woke me up at 4am AGAIN. She was just sitting on my chest staring at me. I love her but she is honestly the most annoying creature on this planet. Anyway, fed her and went back to sleep.", "label": 1, "type": "humanized_AI_casual"},

    # --- CASUAL AI (direct chatbot style) --- should be detected as AI
    {"text": "There are several effective strategies for improving your sleep quality. First, maintain a consistent sleep schedule by going to bed and waking up at the same time each day. Second, create a relaxing bedtime routine that includes avoiding screens for at least 30 minutes before bed. Third, ensure your bedroom is dark, quiet, and cool.", "label": 1, "type": "direct_AI"},
    {"text": "The French Revolution was a pivotal period in European history that began in 1789 and fundamentally transformed French society. Key causes included widespread social inequality, financial crisis, and Enlightenment ideals. The revolution progressed through several phases, from the storming of the Bastille to the Reign of Terror, ultimately leading to the rise of Napoleon Bonaparte.", "label": 1, "type": "direct_AI"},
    {"text": "To effectively learn a new programming language, I recommend starting with the fundamentals: variables, data types, control flow, and functions. Practice with small projects and gradually increase complexity. Online resources like documentation, tutorials, and coding challenges can accelerate your learning. Consistency is more important than intensity.", "label": 1, "type": "direct_AI"},

    # --- FORMAL HUMAN (essays, articles) --- should be detected as HUMAN
    {"text": "The implications of quantum computing for cryptography remain hotly debated. While Shor's algorithm theoretically breaks RSA encryption, practical quantum computers with sufficient qubits and error correction are likely decades away. Nevertheless, NIST has already begun standardizing post-quantum cryptographic algorithms, a prudent if expensive precaution.", "label": 0, "type": "formal_human"},
    {"text": "In the aftermath of the 2008 financial crisis, regulatory frameworks underwent significant revision. The Dodd-Frank Act of 2010 introduced sweeping reforms to the U.S. financial system, including the creation of the Consumer Financial Protection Bureau and new restrictions on proprietary trading by banks.", "label": 0, "type": "formal_human"},
    {"text": "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine. However, the path from laboratory observation to mass production was neither straightforward nor quick. It required the collaborative efforts of Howard Florey and Ernst Boris Chain, along with American pharmaceutical companies, to develop penicillin into a viable therapeutic agent during World War II.", "label": 0, "type": "formal_human"},
    {"text": "Coral reef ecosystems are among the most biodiverse habitats on Earth, supporting approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor. Rising sea temperatures, ocean acidification, and pollution from agricultural runoff pose existential threats to these fragile ecosystems. Restoration efforts, while promising in some localities, cannot yet match the pace of degradation.", "label": 0, "type": "formal_human"},

    # --- CASUAL HUMAN (social media, blogs) --- should be detected as HUMAN
    {"text": "just got back from the dentist and my entire face is numb. tried to drink water and it just dribbled down my chin. my coworkers are having a great time watching me try to eat lunch", "label": 0, "type": "casual_human"},
    {"text": "ok so I finally watched that show everyone was talking about and it was fine? Like not bad but not the life-changing experience twitter promised me. The third episode was actually pretty good though", "label": 0, "type": "casual_human"},
    {"text": "My 6 year old asked me why the sky is blue and I panicked and said because thats its favorite color and now she is telling everyone at school that the sky chose to be blue and her teacher emailed me", "label": 0, "type": "casual_human"},
    {"text": "Been hiking the PCT for 3 weeks now. Feet are destroyed, pack is too heavy, and I ran out of peanut butter yesterday. But watching the sunrise over the Sierra Nevada this morning made it all worth it. Well, almost.", "label": 0, "type": "casual_human"},

    # --- MIXED STYLE (tricky) ---
    {"text": "Climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus, supported by data from multiple independent research institutions, indicates that global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels.", "label": 1, "type": "formal_AI"},
    {"text": "The restaurant was honestly pretty disappointing for the price. We waited 45 minutes for a table despite having a reservation, the appetizers were cold, and the steak was overcooked. The only saving grace was the tiramisu, which was genuinely excellent. Won't be going back.", "label": 0, "type": "review_human"},
    {"text": "Exercise has been shown to have numerous benefits for both physical and mental health. Regular physical activity can help reduce the risk of chronic diseases, improve mood, boost energy levels, and promote better sleep. Even moderate exercise, such as a 30-minute daily walk, can make a significant difference.", "label": 1, "type": "formal_AI"},
    {"text": "Had the worst flight of my life yesterday. Delayed 3 hours, middle seat, crying baby behind me, and they ran out of coffee. The turbulence over the Rockies was genuinely terrifying. At least my luggage made it.", "label": 0, "type": "casual_human"},
]


def evaluate_hard_test(tokenizer, model):
    """Evaluate on 20 handcrafted adversarial samples."""
    print(f"\n{'='*80}")
    print(f"HARD TEST ({len(HARD_SAMPLES)} adversarial samples)")
    print(f"{'='*80}")

    texts = [s["text"] for s in HARD_SAMPLES]
    labels = np.array([s["label"] for s in HARD_SAMPLES])
    preds, probs = predict_batch(tokenizer, model, texts)

    print(f"\n{'Type':<22s} {'True':<6s} {'Pred':<6s} {'AI_prob':<8s} {'OK?':<6s} Text preview")
    print("-" * 120)

    types = {}
    for i, s in enumerate(HARD_SAMPLES):
        true_str = "AI" if s["label"] == 1 else "HUMAN"
        pred_str = "AI" if preds[i] == 1 else "HUMAN"
        is_correct = preds[i] == s["label"]
        mark = "OK" if is_correct else "WRONG"
        print(f'{s["type"]:<22s} {true_str:<6s} {pred_str:<6s} {probs[i]:<8.4f} {mark:<6s} {s["text"][:55]}')

        t = s["type"]
        if t not in types:
            types[t] = {"correct": 0, "total": 0}
        types[t]["total"] += 1
        if is_correct:
            types[t]["correct"] += 1

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    print(f"\n  Hard test accuracy: {int(acc * len(labels))}/{len(labels)} ({acc*100:.0f}%)")
    print(f"  F1: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")

    print(f"\n  Breakdown by type:")
    for t, v in sorted(types.items()):
        print(f"    {t:<22s}: {v['correct']}/{v['total']}")

    # Breakdown: AI detection rate vs Human detection rate
    ai_mask = labels == 1
    human_mask = labels == 0
    ai_recall = accuracy_score(labels[ai_mask], preds[ai_mask]) if ai_mask.any() else 0
    human_recall = accuracy_score(labels[human_mask], preds[human_mask]) if human_mask.any() else 0
    print(f"\n  AI detection rate (true AI → pred AI):       {ai_recall*100:.0f}%")
    print(f"  Human detection rate (true Human → pred Human): {human_recall*100:.0f}%")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="datasets/processed_v2",
                        help="Path to dataset directory with test.parquet")
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    test_metrics = evaluate_test_split(tokenizer, model, args.data_dir)
    hard_metrics = evaluate_hard_test(tokenizer, model)

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Test split:  F1={test_metrics['f1']:.4f}  AUROC={test_metrics['auroc']:.4f}  Acc={test_metrics['accuracy']:.4f}")
    print(f"  Hard test:   F1={hard_metrics['f1']:.4f}  Acc={hard_metrics['accuracy']:.4f}")
    print(f"  Production-ready: {'YES' if test_metrics['f1'] > 0.75 and hard_metrics['accuracy'] > 0.65 else 'NO'}")


if __name__ == "__main__":
    main()
