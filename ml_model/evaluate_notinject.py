
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

print("Loading NotInject dataset...")

dataset = load_dataset("leolee99/NotInject")

df = pd.concat(
    [dataset[split].to_pandas() for split in dataset.keys()],
    ignore_index=True
)

print("Dataset loaded:", len(df), "samples")
print("Columns:", df.columns.tolist())


print("Loading classifier...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]

    return label, float(probs[pred_id])

print("Evaluating prompts...")

predictions = []
confidence = []

for i, prompt in enumerate(df["prompt"]):
    label, score = predict(prompt)
    predictions.append(label)
    confidence.append(score)

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)}")

df["prediction"] = predictions
df["confidence"] = confidence

# NotInject contains only benign examples
df["actual"] = "SAFE"

# Calculate false positive rate
injection_count = sum(
    label == "INJECTION" for label in predictions
)

false_positive_rate = injection_count / len(df)
benign_acceptance_rate = 1 - false_positive_rate

print("\n========== RESULTS ==========")
print("Total samples:", len(df))
print("False positives:", injection_count)
print("False Positive Rate:",
      round(false_positive_rate * 100, 2), "%")
print("Benign Acceptance Rate:",
      round(benign_acceptance_rate * 100, 2), "%")

# Save results
df.to_csv("notinject_results.csv", index=False)

print("\nResults saved to notinject_results.csv")
