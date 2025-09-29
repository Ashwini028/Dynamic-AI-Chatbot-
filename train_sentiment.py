import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# 1. Load dataset
df = pd.read_csv("data/IMDB Dataset.csv").sample(5000)  # reduce size for faster training

# 2. Map sentiment to integers
label2id = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label2id)

# 3. Create HuggingFace dataset
dataset = Dataset.from_pandas(df[["review", "label"]].rename(columns={"review": "text"}))

# 4. Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(tokenize, batched=True)

# 5. Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="models/sentiment_model",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    logging_dir="logs/sentiment_logs",
    save_strategy="no"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded,
    eval_dataset=encoded,  # 👈 for demo, using same dataset
)

# 8. Train
trainer.train()

# 9. Save model
model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

print("✅ Sentiment model training finished and saved in models/sentiment_model/")

