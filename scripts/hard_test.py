"""Hard test set for model validation — 20 adversarial samples."""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "ai/text_detector/checkpoints/transformer/phase1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

HARD_SAMPLES = [
    # --- PARAPHRASED AI (humanized) --- should be detected as AI
    {
        "text": "So basically, yoga is, like, really good for you? It helps with flexibility and stuff, and honestly my anxiety has gotten so much better since I started doing it three months ago. The breathing exercises are kinda weird at first but they actually work lol.",
        "label": 1, "type": "humanized_AI",
    },
    {
        "text": "I tried making sourdough last weekend and it was... okay? Not great. The crust was too thick and the crumb was dense. My friend Sarah said I probably overproofed it but idk. Going to try again this Saturday with a different hydration level.",
        "label": 1, "type": "humanized_AI",
    },
    {
        "text": "ngl the new iPhone is pretty solid. camera upgrade is legit, battery life is better than my old one. only downside is the price tbh. if ur thinking about upgrading from a 14 or older its probably worth it",
        "label": 1, "type": "humanized_AI_casual",
    },
    {
        "text": "The meeting ran way longer than expected. Bob from accounting kept going on about Q3 projections and honestly nobody was paying attention after the first 20 minutes. We need to implement a strict 30-minute cap on these things.",
        "label": 1, "type": "humanized_AI",
    },
    {
        "text": "My cat woke me up at 4am AGAIN. She was just sitting on my chest staring at me. I love her but she is honestly the most annoying creature on this planet. Anyway, fed her and went back to sleep.",
        "label": 1, "type": "humanized_AI_casual",
    },

    # --- CASUAL AI (direct chatbot style) --- should be detected as AI
    {
        "text": "There are several effective strategies for improving your sleep quality. First, maintain a consistent sleep schedule by going to bed and waking up at the same time each day. Second, create a relaxing bedtime routine that includes avoiding screens for at least 30 minutes before bed. Third, ensure your bedroom is dark, quiet, and cool.",
        "label": 1, "type": "direct_AI",
    },
    {
        "text": "The French Revolution was a pivotal period in European history that began in 1789 and fundamentally transformed French society. Key causes included widespread social inequality, financial crisis, and Enlightenment ideals. The revolution progressed through several phases, from the storming of the Bastille to the Reign of Terror, ultimately leading to the rise of Napoleon Bonaparte.",
        "label": 1, "type": "direct_AI",
    },
    {
        "text": "To effectively learn a new programming language, I recommend starting with the fundamentals: variables, data types, control flow, and functions. Practice with small projects and gradually increase complexity. Online resources like documentation, tutorials, and coding challenges can accelerate your learning. Consistency is more important than intensity.",
        "label": 1, "type": "direct_AI",
    },

    # --- FORMAL HUMAN (essays, articles) --- should be detected as HUMAN
    {
        "text": "The implications of quantum computing for cryptography remain hotly debated. While Shor's algorithm theoretically breaks RSA encryption, practical quantum computers with sufficient qubits and error correction are likely decades away. Nevertheless, NIST has already begun standardizing post-quantum cryptographic algorithms, a prudent if expensive precaution.",
        "label": 0, "type": "formal_human",
    },
    {
        "text": "In the aftermath of the 2008 financial crisis, regulatory frameworks underwent significant revision. The Dodd-Frank Act of 2010 introduced sweeping reforms to the U.S. financial system, including the creation of the Consumer Financial Protection Bureau and new restrictions on proprietary trading by banks.",
        "label": 0, "type": "formal_human",
    },
    {
        "text": "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine. However, the path from laboratory observation to mass production was neither straightforward nor quick. It required the collaborative efforts of Howard Florey and Ernst Boris Chain, along with American pharmaceutical companies, to develop penicillin into a viable therapeutic agent during World War II.",
        "label": 0, "type": "formal_human",
    },
    {
        "text": "Coral reef ecosystems are among the most biodiverse habitats on Earth, supporting approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor. Rising sea temperatures, ocean acidification, and pollution from agricultural runoff pose existential threats to these fragile ecosystems. Restoration efforts, while promising in some localities, cannot yet match the pace of degradation.",
        "label": 0, "type": "formal_human",
    },

    # --- CASUAL HUMAN (social media, blogs) --- should be detected as HUMAN
    {
        "text": "just got back from the dentist and my entire face is numb. tried to drink water and it just dribbled down my chin. my coworkers are having a great time watching me try to eat lunch",
        "label": 0, "type": "casual_human",
    },
    {
        "text": "ok so I finally watched that show everyone was talking about and it was fine? Like not bad but not the life-changing experience twitter promised me. The third episode was actually pretty good though",
        "label": 0, "type": "casual_human",
    },
    {
        "text": "My 6 year old asked me why the sky is blue and I panicked and said because thats its favorite color and now she is telling everyone at school that the sky chose to be blue and her teacher emailed me",
        "label": 0, "type": "casual_human",
    },
    {
        "text": "Been hiking the PCT for 3 weeks now. Feet are destroyed, pack is too heavy, and I ran out of peanut butter yesterday. But watching the sunrise over the Sierra Nevada this morning made it all worth it. Well, almost.",
        "label": 0, "type": "casual_human",
    },

    # --- MIXED STYLE (tricky) ---
    {
        "text": "Climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus, supported by data from multiple independent research institutions, indicates that global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels.",
        "label": 1, "type": "formal_AI",
    },
    {
        "text": "The restaurant was honestly pretty disappointing for the price. We waited 45 minutes for a table despite having a reservation, the appetizers were cold, and the steak was overcooked. The only saving grace was the tiramisu, which was genuinely excellent. Won't be going back.",
        "label": 0, "type": "review_human",
    },
    {
        "text": "Exercise has been shown to have numerous benefits for both physical and mental health. Regular physical activity can help reduce the risk of chronic diseases, improve mood, boost energy levels, and promote better sleep. Even moderate exercise, such as a 30-minute daily walk, can make a significant difference.",
        "label": 1, "type": "formal_AI",
    },
    {
        "text": "Had the worst flight of my life yesterday. Delayed 3 hours, middle seat, crying baby behind me, and they ran out of coffee. The turbulence over the Rockies was genuinely terrifying. At least my luggage made it.",
        "label": 0, "type": "casual_human",
    },
]

print(f"=== HARD TEST SET ({len(HARD_SAMPLES)} samples) ===")
print(f"{'Type':<22s} {'True':<6s} {'Pred':<6s} {'Score':<8s} {'OK?':<6s} Text preview")
print("-" * 120)

correct = 0
types = {}
for s in HARD_SAMPLES:
    inputs = tokenizer(s["text"], return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = int(probs[0, 1] > 0.5)
        ai_prob = float(probs[0, 1])

    is_correct = pred_label == s["label"]
    correct += int(is_correct)
    true_str = "AI" if s["label"] == 1 else "HUMAN"
    pred_str = "AI" if pred_label == 1 else "HUMAN"
    mark = "OK" if is_correct else "WRONG"
    print(f'{s["type"]:<22s} {true_str:<6s} {pred_str:<6s} {ai_prob:<8.4f} {mark:<6s} {s["text"][:55]}')

    t = s["type"]
    if t not in types:
        types[t] = {"correct": 0, "total": 0}
    types[t]["total"] += 1
    if is_correct:
        types[t]["correct"] += 1

print(f"\nHard test accuracy: {correct}/{len(HARD_SAMPLES)} ({100*correct/len(HARD_SAMPLES):.0f}%)")
print(f"\nBreakdown by type:")
for t, v in types.items():
    print(f"  {t:<22s}: {v['correct']}/{v['total']}")
