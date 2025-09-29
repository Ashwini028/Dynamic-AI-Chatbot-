import streamlit as st
import requests

st.set_page_config(page_title="Dynamic AI Chatbot", page_icon="🤖", layout="centered")

st.title(" Dynamic AI Chatbot")
st.write("Chat with your trained chatbot (via FastAPI backend).")

# User session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input(" You:", "", key="user_input")

if st.button("Send"):
    if user_input.strip() != "":
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Send request to backend FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                params={"query": user_input, "user": "web_user"}
            )
            if response.status_code == 200:
                bot_reply = response.json().get("response", "️ No response")
            else:
                bot_reply = f"️ Error: {response.status_code}"

        except Exception as e:
            bot_reply = f" Backend not running: {e}"

        # Save bot reply
        st.session_state.messages.append({"role": "bot", "content": bot_reply})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f" **You:** {msg['content']}")
    else:
        st.markdown(f" **Bot:** {msg['content']}")
