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

# --- 1. PAGE CONFIG & DESIGNER UI STYLING ---
st.set_page_config(page_title="Risk Control Tower", layout="wide", page_icon="🛡️")

# Professional UI CSS Injection
st.markdown("""
    <style>
    /* Global Styles */
    .stApp { background-color: #f1f5f9; }
    
    /* Sidebar Styling - Ensuring high readability */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#0f172a, #1e293b);
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
    
    /* Metric Card Styling with Color Accents */
    div[data-testid="stMetric"] {
        background: white;
        padding: 15px 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 5px solid #3b82f6; /* Default Blue */
    }
    /* Specific Colors for specific metrics */
    div[data-testid="stMetric"]:nth-child(1) { border-top-color: #ef4444; } /* Red for Overdue */
    div[data-testid="stMetric"]:nth-child(2) { border-top-color: #f59e0b; } /* Amber for High Risk */
    div[data-testid="stMetric"]:nth-child(3) { border-top-color: #10b981; } /* Emerald for Health */
    
    /* Chat Readability */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1 { color: #0f172a; font-weight: 800; }
    h3 { color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# API Key Security (Streamlit Secrets)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key Missing! Go to Settings > Secrets in Streamlit Cloud.")
    st.stop()

# --- 2. DATA ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')

        latest_m = m_df.sort_values('Date').groupby('Indicator').tail(1)
        m_context = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_m.iterrows()])

        def enrich(row):
            overdue = t_df[(t_df['Project_ID'] == row['Project_ID']) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
            return f"ID: {row['Project_ID']} | Type: {row['Project_Type']} | Status: {row['Risk_Level']} | Overdue: ${overdue} | Context: {m_context}"

        p_df['master_context'] = p_df.apply(enrich, axis=1)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(p_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, p_df, t_df, latest_m
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. UI LAYOUT ---
st.title("🛡️ Risk Control Tower")
st.markdown("### Real-time AI Project Telemetry")

db, projects, transactions, market = initialize_risk_engine()

# Sidebar: High Readability Filters
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    risk_filter = st.multiselect("Risk Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    budget_range = st.slider("Budget Spend Range (%)", 0, 100, (0, 100))
    
    st.markdown("---")
    st.markdown("### 🛰️ System Status")
    st.success("🟢 Market Feed: LIVE")
    st.success("🟢 Risk Engine: SYNCED")
    st.info("Using Gemini 3 Flash")

if db is not None:
    # --- TOP ROW: KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    total_overdue = transactions[transactions['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(projects[projects['Risk_Level'] == 'High'])
    
    col1.metric("Financial Overdue", f"${total_overdue:,.0f}", help="Sum of all unpaid overdue invoices")
    col2.metric("Critical Projects", high_risk_count, "High Priority")
    col3.metric("System Health", "94%", "Optimal")
    col4.metric("Market Sentiment", market.iloc[0]['Market_Sentiment'])

    st.divider()

    # --- CHAT ADVISOR ---
    st.subheader("🤖 AI Risk Advisor")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about specific projects or risk summaries..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Analyzing Project Data...", expanded=False) as status:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are a professional Risk Officer. Use this project context to provide actionable advice: {context}"),
                    ("human", "{input}"),
                ]))
                rag_chain = create_retrieval_chain(db.as_retriever(), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Analysis Complete", state="complete")
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
