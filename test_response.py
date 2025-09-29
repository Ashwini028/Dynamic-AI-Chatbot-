import sys, os

# --- Force Python to treat dynamic_chatbot as project root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.response_generator import generate_response

sample_texts = [
    "Book me a flight to London",
    "I am so happy today!",
    "Who is the president of USA?"
]

for text in sample_texts:
    print("\nUSER:", text)
    print("BOT :", generate_response(text))
