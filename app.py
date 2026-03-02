import streamlit as st
import pandas as pd
import os
import time

# 2026 Core Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Risk Control Tower", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #1e293b; color: white; }
    .stChatFloatingInputContainer { background-color: #ffffff; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# API Key Check
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key not found in Secrets.")
    st.stop()

# --- 2. DATA ENGINE (The Brain) ---
@st.cache_resource
def initialize_risk_engine():
    try:
        # Load Datasets
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')

        # Market logic
        latest_m = m_df.sort_values('Date').groupby('Indicator').tail(1)
        m_context = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_m.iterrows()])

        def enrich(row):
            overdue = t_df[(t_df['Project_ID'] == row['Project_ID']) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
            return f"ID: {row['Project_ID']} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | Overdue: ${overdue} | Market: {m_context}"

        p_df['master_context'] = p_df.apply(enrich, axis=1)
        
        # Vectorization
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(p_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, p_df, t_df, latest_m
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. UI LAYOUT ---
st.title("🛡️ AI Project Risk Control Tower")

# Sidebar for Group Project Options
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1162/1162456.png", width=100)
    st.header("Project Filters")
    risk_filter = st.multiselect("Filter by Risk Level", ["High", "Medium", "Low"], default=["High", "Medium"])
    budget_range = st.slider("Budget Utilization %", 0, 100, (0, 100))
    st.divider()
    st.info("💡 **Tip:** Ask the AI to 'Compare project X and Y' for deep analysis.")

# Load Data and Show Metrics
db, projects, transactions, market = initialize_risk_engine()

if db is not None:
    # 📈 TOP ROW METRICS
    col1, col2, col3, col4 = st.columns(4)
    total_overdue = transactions[transactions['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(projects[projects['Risk_Level'] == 'High'])
    
    col1.metric("Total Overdue", f"${total_overdue:,.0f}", delta="-2.3% (Weekly)")
    col2.metric("High Risk Projects", high_risk_count, delta="↑ 2", delta_color="inverse")
    col3.metric("Avg Complexity", f"{projects['Complexity_Score'].mean():.1f}/10")
    col4.metric("Market Sentiment", market.iloc[0]['Market_Sentiment'])

    # 💬 CHAT INTERFACE
    st.subheader("🤖 Risk Advisory Agent")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about project health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Searching project memory...", expanded=False) as status:
                st.write("Fetching financial records...")
                time.sleep(0.5)
                st.write("Analyzing market trends...")
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are a professional Risk Officer. Answer based on context: {context}"),
                    ("human", "{input}"),
                ]))
                
                rag_chain = create_retrieval_chain(db.as_retriever(), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Report Complete!", state="complete", expanded=False)
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
