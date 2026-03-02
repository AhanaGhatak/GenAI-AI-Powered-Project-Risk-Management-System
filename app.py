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

# --- 1. PAGE CONFIG & ENHANCED UI STYLING ---
st.set_page_config(page_title="Risk Control Tower", layout="wide", page_icon="🛡️")

# Custom CSS for Visibility and Modern Aesthetics
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8fafc; }
    
    /* Sidebar Text Visibility Fix */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important; /* Deep Navy */
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #f1f5f9 !important; /* Off-White for readability */
        font-weight: 500;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] { color: #1e293b; font-weight: 700; }
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Chat Header Styling */
    .chat-header {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# API Key Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key not found in Secrets. Please add it to Streamlit Cloud Settings.")
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
            return f"ID: {row['Project_ID']} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | Overdue: ${overdue} | Market: {m_context}"

        p_df['master_context'] = p_df.apply(enrich, axis=1)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(p_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, p_df, t_df, latest_m
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. UI LAYOUT ---
st.title("🛡️ AI Project Risk Control Tower")
st.markdown("---")

db, projects, transactions, market = initialize_risk_engine()

# Sidebar with Fixed Visibility
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### Filter Telemetry")
    risk_filter = st.multiselect("Risk Sensitivity", ["High", "Medium", "Low"], default=["High", "Medium"])
    budget_range = st.slider("Budget Utilization Target (%)", 0, 100, (0, 100))
    
    st.markdown("---")
    st.markdown("### 🤖 Agent Status")
    st.success("● Market Agent: Active")
    st.success("● Risk Agent: Active")
    st.warning("● Prediction: Dynamic")

if db is not None:
    # Top Row Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    total_overdue = transactions[transactions['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(projects[projects['Risk_Level'] == 'High'])
    
    col1.metric("Financial Overdue", f"${total_overdue:,.0f}", delta="-2.3%")
    col2.metric("Critical Alerts", high_risk_count, delta="High Risk", delta_color="inverse")
    col3.metric("System Health", "94%", delta="Optimal")
    col4.metric("Market Index", market.iloc[0]['Market_Sentiment'])

    st.markdown("<div class='chat-header'>💬 Advisor Chat</div>", unsafe_allow_html=True)
    
    # Chat History logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about risk patterns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=False) as status:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are a professional Risk Officer. Use context: {context}"),
                    ("human", "{input}"),
                ]))
                rag_chain = create_retrieval_chain(db.as_retriever(), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Analysis Complete", state="complete")
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
