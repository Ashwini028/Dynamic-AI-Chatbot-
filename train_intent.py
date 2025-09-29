import json
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# ---------------------------
# 1. Load JSON dataset
# ---------------------------
with open(r"D:\AI Chatbot Project\data\data_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level keys in JSON:", data.keys())

# Pick train split (adjust if your JSON structure is different)
train_data = data["train"]
print("First 5 rows from train_data:", train_data[:5])

# Convert into DataFrame
df = pd.DataFrame(train_data, columns=["sentence", "intent"])
print("DataFrame preview:")
print(df.head())
print("Number of rows:", len(df))

# ---------------------------
# 2. Preprocess dataset
# ---------------------------
# Map intent labels to integers
unique_intents = sorted(df["intent"].unique())
label2id = {label: i for i, label in enumerate(unique_intents)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["intent"].map(label2id)

# HuggingFace Dataset
dataset = Dataset.from_pandas(df[["sentence", "label"]])

# ---------------------------
# 3. Tokenizer
# ---------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=64)

encoded_dataset = dataset.map(tokenize, batched=True)

# ---------------------------
# 4. Model
# ---------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(unique_intents),
    id2label=id2label,
    label2id=label2id
)

# ---------------------------
# 5. Training
# ---------------------------
training_args = TrainingArguments(
    output_dir="models/intent_model",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_dir="logs/intent_logs",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
)

# ... your training code ...
trainer.train()

# ---------------------------
# 6. Save model
# ---------------------------
model.save_pretrained("models/intent_model")
tokenizer.save_pretrained("models/intent_model")

print("✅ Intent model saved in models/intent_model/")

