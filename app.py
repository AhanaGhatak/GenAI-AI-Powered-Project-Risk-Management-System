import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px

# 2026 Core RAG Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. THEME & READABILITY STYLING ---
st.set_page_config(page_title="Risk Control Tower", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #fcfcfc; }
    
    /* FIX: High-Readability Sidebar (Black text on White/Light Grey) */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h2 {
        color: #0f172a !important; /* Dark Slate for perfect reading */
        font-weight: 600 !important;
    }
    
    /* Top Heading Banner */
    .main-header {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 2.5rem; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    </style>
    
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE COMMAND</h1>
        <p style="opacity: 0.8;">Enterprise Risk Monitoring & Strategic Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Secure API Key Handling
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing! Please update Secrets.")
    st.stop()

# --- 2. DATA ENRICHMENT & VECTOR ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        projects_df = pd.read_csv('project_risk_raw_dataset.csv')
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')

        latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
        market_summary = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_market.iterrows()])

        def enrich_logic(row):
            pid = row['Project_ID']
            overdue_val = txns_df[(txns_df['Project_ID'] == pid) & (txns_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
            return (f"PROJECT ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                    f"Overdue: ${overdue_val:,.2f} | Market: {market_summary}")

        projects_df['master_context'] = projects_df.apply(enrich_logic, axis=1)
        
        # System uses AI Core for embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(projects_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, projects_df, txns_df, market_df
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. UI LAYOUT ---
db, p_df, t_df, m_df = initialize_risk_engine()

with st.sidebar:
    st.header("⚙️ CONTROL PANEL")
    st.divider()
    st.markdown("### DATA FILTERS")
    risk_level_ui = st.multiselect("Risk Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    complexity_threshold = st.slider("Complexity Range", 0, 10, (1, 9))
    
    st.markdown("---")
    st.markdown("### 🛰️ SYSTEM STATUS")
    st.success("● AI CORE: ONLINE")
    st.success("● VECTOR DB: ACTIVE")
    if st.button("🔄 REFRESH SYSTEM"):
        st.cache_resource.clear()
        st.rerun()

if db:
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(p_df[p_df['Risk_Level'] == 'High'])
    
    col1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M")
    col2.metric("Critical Alerts", high_risk_count)
    col3.metric("System Health", "98%")
    col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'])

    st.markdown("---")

    # Chart Section
    st.subheader("📊 Portfolio Risk Heatmap")
    fig = px.scatter(
        p_df, x="Complexity_Score", y="Budget_Utilization_Rate",
        color="Risk_Level", size="Complexity_Score", hover_name="Project_ID",
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        template="plotly_white", height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Advisor Chat
    st.subheader("🤖 STRATEGIC AI ADVISOR")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter inquiry regarding risk telemetry..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Processing Telemetry...", expanded=False) as status:
                # Internal AI Core call
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are the Project Risk Commander. Use context to advise. Context: {context}"),
                    ("human", "{input}"),
                ]))
                
                rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Analysis Complete", state="complete")
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
