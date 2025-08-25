import streamlit as st
import joblib
import pandas as pd
import requests

# Load model
model = joblib.load("transfer_fee_model.pkl")

# Ollama config
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"

# Page setup
st.set_page_config(page_title="⚽ Football Assistant", page_icon="⚽", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stTextInput > div > input {
            background-color: #ffffff;
        }
        .stButton>button {
            background-color: #0057e7;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("## ⚽ Football Assistant")
st.markdown("Welcome to your all-in-one football assistant website! Use the tabs below to predict transfer fees or chat with an AI football expert.")

# Tabs
tab1, tab2 = st.tabs([
    "💰 Predict Transfer Fees", 
    "💬 Football Chatbot"
])

# ---------------------- TAB 1 ----------------------
with tab1:
    st.markdown("### 📊 Transfer Fee Predictor")
    st.info("Fill in the details of the player to estimate their transfer market value.")

    col1, col2 = st.columns(2)

    with col1:
        season = st.number_input("📅 Season", min_value=2000, max_value=2025, value=2024)
        window = st.selectbox("🪟 Transfer Window", ["Summer", "Winter"])
        player_age = st.slider("🎂 Player Age", 16, 40, 25)
        player_pos = st.selectbox("🧭 Player Position", ["GK", "DF", "MF", "FW"])

    with col2:
        player_nation = st.text_input("🏳️ Player Nation", "Germany")
        market_val_amnt = st.number_input("💸 Market Value (€)", value=20000000)
        is_free = st.checkbox("🆓 Free Transfer?")
        is_loan = st.checkbox("🔄 Is Loan?")
        is_loan_end = st.checkbox("🏁 Is Loan End?")
        is_retired = st.checkbox("⚰️ Is Retired?")

    if st.button("🚀 Predict Transfer Fee"):
        input_data = pd.DataFrame([{
            "season": season,
            "window": window,
            "player_age": player_age,
            "player_nation": player_nation,
            "player_pos": player_pos,
            "market_val_amnt": market_val_amnt,
            "is_free": int(is_free),
            "is_loan": int(is_loan),
            "is_loan_end": int(is_loan_end),
            "is_retired": int(is_retired)
        }])

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"🎯 Estimated Transfer Fee: **€{prediction:,.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ---------------------- TAB 2 ----------------------
with tab2:
    st.markdown("### 🤖 Football Chatbot")
    st.info("Ask anything about football — stats, players, transfers, clubs, and history!")

    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a football expert assistant. "
                    "Only answer football-related questions about players, teams, stats, transfers, history, and tournaments. "
                    "Be brief and factual unless the user asks for more detail."
                )
            }
        ]

    # Chat history
    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask a football-related question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            response = requests.post(OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "messages": st.session_state.messages,
                "stream": False
            })
            response.raise_for_status()

            reply = response.json()["message"]["content"]
            st.session_state.messages.append({"role": "assistant", "content": reply})

            with st.chat_message("assistant"):
                st.markdown(reply)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error communicating with Ollama:\n{e}")
