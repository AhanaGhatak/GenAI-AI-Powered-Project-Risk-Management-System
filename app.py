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

# --- 1. PREMIUM UI & BRANDING STYLING ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* Global Background and Typography */
    .stApp { background-color: #f0f2f6; font-family: 'Inter', sans-serif; }
    
    /* SIDEBAR: High-Contrast & Professional */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* TOP BANNER: Modern Gradient */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
    }
    .main-header h1 { color: white !important; font-size: 3rem !important; letter-spacing: -1px; }
    
    /* METRIC CARDS: Glowing Accents */
    div[data-testid="stMetric"] {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border-bottom: 4px solid #e2e8f0;
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-5px); }
    
    /* Color coding for metric status */
    div[data-testid="stMetric"]:nth-child(1) { border-bottom-color: #ff4b4b; } /* Overdue */
    div[data-testid="stMetric"]:nth-child(2) { border-bottom-color: #ffa500; } /* Risk */
    div[data-testid="stMetric"]:nth-child(3) { border-bottom-color: #00d26a; } /* Health */
    div[data-testid="stMetric"]:nth-child(4) { border-bottom-color: #6366f1; } /* Market */

    /* Chat Styling */
    .stChatMessage {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    </style>
    
    <div class="main-header">
        <h1>RISK INTELLIGENCE COMMAND</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Strategic Multi-Agent Oversight & Risk Mitigation</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing! Please update Secrets.")
    st.stop()

# --- 2. DATA ENRICHMENT ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')

        latest_market = m_df.sort_values('Date').groupby('Indicator').tail(1)
        market_summary = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_market.iterrows()])

        def enrich_logic(row):
            pid = row['Project_ID']
            overdue_val = t_df[(t_df['Project_ID'] == pid) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
            return (f"PROJECT ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                    f"Overdue: ${overdue_val:,.2f} | Market: {market_summary}")

        p_df['master_context'] = p_df.apply(enrich_logic, axis=1)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(p_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, p_df, t_df, m_df
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. DASHBOARD EXECUTION ---
db, p_df, t_df, m_df = initialize_risk_engine()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.header("SYSTEM CONTROLS")
    st.divider()
    risk_level_ui = st.multiselect("High-Priority Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    complexity_threshold = st.slider("Complexity Scale", 0, 10, (2, 9))
    
    st.markdown("---")
    st.subheader("🛰️ ENGINE STATUS")
    st.success("● CORE PROCESSOR: ACTIVE")
    st.success("● NEURAL MEMORY: SYNCED")
    if st.button("🚀 REBOOT SYSTEM"):
        st.cache_resource.clear()
        st.rerun()

if db:
    # KPI SECTION
    col1, col2, col3, col4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(p_df[p_df['Risk_Level'] == 'High'])
    
    col1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", "Total Overdue")
    col2.metric("Critical Projects", high_risk_count, "Immediate Attention")
    col3.metric("System Health", "98%", "Stable")
    col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Live Feed")

    st.markdown("---")

    # VISUALIZATION SECTION
    st.subheader("📊 Portfolio Risk Heatmap")
    
    fig = px.scatter(
        p_df, x="Complexity_Score", y="Budget_Utilization_Rate",
        color="Risk_Level", size="Complexity_Score", hover_name="Project_ID",
        color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00d26a"},
        template="plotly_white", height=500
    )
    fig.update_layout(bordercolor="#e2e8f0", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ADVISOR SECTION
    st.subheader("💬 STRATEGIC ADVISORY CHAT")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Inquire about specific project risks..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Consulting Risk Engine...", expanded=False) as status:
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are the Strategic Risk Advisor. Provide concise, professional analysis. Context: {context}"),
                    ("human", "{input}"),
                ]))
                rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Analysis Delivered", state="complete")
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"])
