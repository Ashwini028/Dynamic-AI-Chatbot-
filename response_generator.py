# utils/response_generator.py

class ResponseGenerator:
    def __init__(self, models=None, memory=None):
        self.models = models
        self.memory = memory

    async def generate(self, session_id, user_text, intent, entities, sentiment):
        """
        Generate a reply based on intent, entities, sentiment.
        """
        intent_label = intent.get("label", "unknown")

        # --- Rule-based templates ---
        templates = {
            "greeting": "Hello! How can I help you today?",
            "goodbye": "Goodbye! Have a nice day!",
            "pricing": "Our pricing starts at $9.99/month. Do you want the full list?",
            "refund_policy": "Our refund policy allows returns within 30 days."
        }

        if intent_label in templates:
            return templates[intent_label]

        # --- Sentiment-aware fallback ---
        if sentiment.get("sentiment") == "negative":
            return "I’m sorry if something went wrong. Can you explain more so I can help?"

        # --- Default fallback ---
        return "Sorry, I didn’t quite understand. Could you rephrase?"


def generate_response():
    return None