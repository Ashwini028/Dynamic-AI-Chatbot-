import sys, os

# --- Force Python to treat dynamic_chatbot as project root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.preprocess import load_models, predict_intent, predict_ner, predict_sentiment

# Load models
intent_model, intent_tokenizer, ner_model, ner_tokenizer, sent_model, sent_tokenizer = load_models()

# Test inputs
sample_text = "Book me a flight to New York tomorrow"

print("\n=== INTENT PREDICTION ===")
print(predict_intent(sample_text, intent_model, intent_tokenizer))

print("\n=== NER PREDICTION ===")
print(predict_ner(sample_text, ner_model, ner_tokenizer))

print("\n=== SENTIMENT PREDICTION ===")
print(predict_sentiment("I love this chatbot!", sent_model, sent_tokenizer))
