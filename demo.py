from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

MODEL_NAME = "ProsusAI/finbert"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
mdl.eval()

print("id2label:", mdl.config.id2label)

headline = "Apple stock rises after strong earnings"
enc = tok(headline, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = mdl(**enc).logits
probs = torch.softmax(logits, dim=-1).numpy()[0]

order = [mdl.config.id2label[i].lower() for i in range(len(probs))]   # e.g. ['negative','neutral','positive']
prob_map = dict(zip(order, probs))
neg = prob_map.get("negative", 0.0)
neu = prob_map.get("neutral", 0.0)
pos = prob_map.get("positive", 0.0)
compound = pos - neg

print(f"Headline: {headline}")
print(f"FinBERT Sentiment Scores:\n  neg={neg:.4f}, neu={neu:.4f}, pos={pos:.4f}, compound={compound:+.4f}")
