# legacy, superseded by src/marathi_llm/eval/
"""
evaluate.py — Quantitative evaluation using chrF++ score
---------------------------------------------------------
Requires:  pip install sacrebleu
Requires:  evals/comparison.csv (run infer.py first)

Usage:  python evaluate.py
Output: evals/scores.csv  +  terminal summary table
"""

import os, csv
from sacrebleu.metrics import CHRF

# ── Reference answers (Marathi ground truth) ─────────────────
# These correspond 1-to-1 with PROMPTS in infer.py
REFERENCES = [
    "1. संतुलित आहार घ्या आणि भरपूर फळे आणि भाज्या खा. 2. नियमित व्यायाम करा. 3. पुरेशी झोप घ्या.",
    "लाल, निळा आणि पिवळा असे तीन प्राथमिक रंग आहेत.",
    "पाणी जीवनासाठी अत्यंत आवश्यक आहे. ते शरीरातील पेशींना कार्यक्षम ठेवते आणि पचनक्रिया सुधारते.",
    "भारताची राजधानी नवी दिल्ली आहे.",
    "सूर्यमाला म्हणजे सूर्य आणि त्याच्याभोवती फिरणारे ग्रह, उपग्रह, लघुग्रह आणि धूमकेतू यांचा समूह.",
    "प्रिय मित्र, मी तुला हे पत्र लिहित आहे कारण मला तुझ्याशी काही महत्त्वाच्या गोष्टी सांगायच्या आहेत.",
    "झाडे आपल्याला ऑक्सिजन देतात, पर्यावरण स्वच्छ ठेवतात आणि पाऊस आणण्यास मदत करतात.",
    "संगणक एक इलेक्ट्रॉनिक उपकरण आहे जे माहिती साठवते, प्रक्रिया करते आणि विविध कामांसाठी वापरले जाते.",
    "वेळेचे व्यवस्थापन करण्यासाठी दैनंदिन वेळापत्रक बनवा, प्राधान्यक्रम ठरवा आणि विलंब टाळा.",
    "चांगला विद्यार्थी नियमित अभ्यास करतो, शिक्षकांचा आदर करतो आणि जिज्ञासू वृत्ती ठेवतो.",
]

COMPARISON_PATH = "evals/comparison.csv"
SCORES_PATH     = "evals/scores.csv"

if not os.path.exists(COMPARISON_PATH):
    print(f"Error: {COMPARISON_PATH} not found. Run infer.py first.")
    exit(1)

# ── Load comparison CSV ───────────────────────────────────────
prompts, base_outputs, ft_outputs = [], [], []
with open(COMPARISON_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        prompts.append(row["prompt"])
        base_outputs.append(row["base_output"])
        ft_outputs.append(row["finetuned_output"])

if len(prompts) != len(REFERENCES):
    print(f"Warning: {len(prompts)} rows in CSV but {len(REFERENCES)} references defined.")
    print("Truncating to the shorter list.")
    n = min(len(prompts), len(REFERENCES))
    prompts, base_outputs, ft_outputs = prompts[:n], base_outputs[:n], ft_outputs[:n]
    references = REFERENCES[:n]
else:
    references = REFERENCES

# ── Compute chrF++ per sample ─────────────────────────────────
chrf = CHRF(word_order=2)  # word_order=2 → chrF++

rows = []
for i, (prompt, base, ft, ref) in enumerate(zip(prompts, base_outputs, ft_outputs, references)):
    base_score = chrf.sentence_score(base, [ref]).score
    ft_score   = chrf.sentence_score(ft,   [ref]).score
    delta      = ft_score - base_score
    rows.append({
        "prompt"       : prompt,
        "base_chrf"    : round(base_score, 2),
        "finetuned_chrf": round(ft_score, 2),
        "delta"        : round(delta, 2),
    })

# ── Corpus-level scores ───────────────────────────────────────
corpus_base = chrf.corpus_score(base_outputs, [references]).score
corpus_ft   = chrf.corpus_score(ft_outputs,   [references]).score

# ── Save scores CSV ───────────────────────────────────────────
os.makedirs("evals", exist_ok=True)
with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "base_chrf", "finetuned_chrf", "delta"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Scores saved → {SCORES_PATH}")

# ── Terminal summary ──────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'#':<4} {'Base':>8} {'FineTuned':>10} {'Delta':>8}  Prompt")
print("-" * 70)
for i, row in enumerate(rows):
    arrow = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
    print(f"{i+1:<4} {row['base_chrf']:>8.2f} {row['finetuned_chrf']:>10.2f} "
          f"{arrow}{abs(row['delta']):>6.2f}  {row['prompt'][:40]}")

print("=" * 70)
print(f"\nCorpus chrF++ — Base model    : {corpus_base:.2f}")
print(f"Corpus chrF++ — Fine-tuned    : {corpus_ft:.2f}")
print(f"Overall improvement           : {corpus_ft - corpus_base:+.2f} points")

if corpus_ft > corpus_base:
    print("\n✓ Fine-tuning improved Marathi generation quality.")
else:
    print("\n✗ Fine-tuned model scored lower — check outputs for language mixing.")
