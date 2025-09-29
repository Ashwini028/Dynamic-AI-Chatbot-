import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

# ---------------------------
# 1. Load Models
# ---------------------------
def load_models():
    # Intent model
    intent_model = AutoModelForSequenceClassification.from_pretrained("models/intent_model")
    intent_tokenizer = AutoTokenizer.from_pretrained("models/intent_model")

    # NER model
    ner_model = AutoModelForTokenClassification.from_pretrained("models/ner_model")
    ner_tokenizer = AutoTokenizer.from_pretrained("models/ner_model")

    # Sentiment model
    sent_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_model")
    sent_tokenizer = AutoTokenizer.from_pretrained("models/sentiment_model")

    return intent_model, intent_tokenizer, ner_model, ner_tokenizer, sent_model, sent_tokenizer


# ---------------------------
# 2. Intent Prediction
# ---------------------------
def predict_intent(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).item()

    # If labels.json exists, use it, else return raw ID
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    return id2label.get(preds, str(preds)) if id2label else str(preds)


# ---------------------------
# 3. NER Prediction
# ---------------------------
def predict_ner(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = []
    for token, pred in zip(tokens, predictions[0].numpy()):
        if token.startswith("##"):  # skip subwords
            continue
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
        labels.append((token, id2label.get(pred, str(pred))))
    return labels


# ---------------------------
# 4. Sentiment Prediction
# ---------------------------
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).item()

    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    return id2label.get(preds, str(preds)) if id2label else str(preds)
