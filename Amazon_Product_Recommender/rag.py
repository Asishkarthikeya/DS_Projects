import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent

st.sidebar.title("🔑 Enter GROQ API Key")
GROQ_API_KEY = st.sidebar.text_input(
    "GROQ API Key",
    type="password",
    help="Paste your full key (including the leading 'gsk_…')"
)
if not GROQ_API_KEY:
    st.sidebar.warning("Please enter your API key to continue.")
    st.stop()

if "agent" not in st.session_state:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        temperature=0
    )
    agent = create_csv_agent(
        llm,
        "All_Categories.csv",
        verbose=True,
        allow_dangerous_code=True
    )
    st.session_state.agent = agent
    st.session_state.history = []  

agent = st.session_state.agent

st.title("🔍 RAG Chat with the Product Search")
st.markdown(
    """
    Ask natural‑language questions about the Products. It will be retrieved based on your natural language query.
    """
)

if st.sidebar.checkbox("Show raw data"):
    df = pd.read_csv("All_Categories.csv")
    st.dataframe(df)

for turn in st.session_state.history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
    st.write("---")

query = st.text_input("Enter your question here:")
if st.button("Send"):
    with st.spinner("Thinking…"):
        try:
            if st.session_state.history:
                history_str = "\n".join(
                    f"User: {h['user']}\nAssistant: {h['assistant']}"
                    for h in st.session_state.history
                )
                prompt = f"{history_str}\nUser: {query}"
            else:
                prompt = query

            answer = agent.run(prompt)

        except Exception as e:
            err = str(e)
            if "invalid_api_key" in err.lower():
                st.error("🔐 Authentication failed: Invalid GROQ API Key.")
            else:
                st.error(f"❗ An error occurred:\n{err}")
            st.stop()

    st.session_state.history.append({
        "user": query,
        "assistant": answer
    })
    st.rerun()
