"""
run_demo.py
Quick demo script to verify the Week 1 pipeline works.
Run with: python run_demo.py
"""

import json
from app.analyzer import analyze, analyze_batch, results_to_dataframe

# ── Single text test ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SINGLE TEXT ANALYSIS")
print("="*60)

text = "You are completely useless and nobody wants you around."
result = analyze(text, include_bias=True)

print(f"\nText:       {result['text']}")
print(f"Risk level: {result['risk_level']}")
print(f"Flags:      {result['flags']}")
print(f"\nToxicity scores:")
for k, v in result['toxicity'].items():
    bar = "█" * int(v * 20)
    print(f"  {k:<20} {v:.4f}  {bar}")
print(f"\nSentiment compound: {result['sentiment']['compound']}")
if result['bias']:
    print(f"\nBias scores:")
    for k, v in result['bias'].items():
        print(f"  {k:<20} {v:.4f}")

# ── Batch test from sample data ───────────────────────────────────────────────
print("\n" + "="*60)
print("BATCH ANALYSIS — sample_inputs.json")
print("="*60)

with open("data/sample_inputs.json") as f:
    samples = json.load(f)

texts = [s["text"] for s in samples]
results = analyze_batch(texts, include_bias=False)
df = results_to_dataframe(results)

print("\nResults summary:")
print(df[["text", "risk_level", "toxicity", "sentiment_compound", "flags"]].to_string(index=False))

print("\n" + "="*60)
print("Week 1 pipeline working correctly!")
print("="*60)
