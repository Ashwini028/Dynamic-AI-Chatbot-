# save_pretrained_models.py
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

os.makedirs("models", exist_ok=True)

# -----------------------------
# 1. Intent Classifier (dummy)
# -----------------------------
print("Downloading intent model...")
intent_model_name = "bert-base-uncased"  # placeholder, acts as simple classifier
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name, num_labels=2)
intent_model.save_pretrained("models/intent_model")
intent_tokenizer.save_pretrained("models/intent_model")

# -----------------------------
# 2. NER Model
# -----------------------------
print("Downloading NER model...")
ner_model_name = "dslim/bert-base-NER"  # pretrained NER
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_model.save_pretrained("models/ner_model")
ner_tokenizer.save_pretrained("models/ner_model")

# -----------------------------
# 3. Sentiment Classifier
# -----------------------------
print("Downloading sentiment model...")
sent_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)
sent_model.save_pretrained("models/sentiment_model")
sent_tokenizer.save_pretrained("models/sentiment_model")

print(" All pretrained models downloaded and saved into /models/")


