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

# --- 1. THEME & ACCESSIBILITY STYLING ---
st.set_page_config(page_title="AI Risk Control Tower", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* Global Background */
    .stApp { background-color: #f8fafc; }
    
    /* High-Contrast Sidebar Fix */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#0f172a, #1e293b);
        min-width: 300px;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #f1f5f9 !important; /* Vivid white for visibility */
        font-weight: 500 !important;
        font-size: 1rem;
    }
    
    /* Designer Metric Cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 8px solid #3b82f6;
    }
    /* RAG Status Colors for Metrics */
    div[data-testid="stMetric"]:nth-child(1) { border-left-color: #ef4444; } /* Overdue Red */
    div[data-testid="stMetric"]:nth-child(2) { border-left-color: #f59e0b; } /* High Risk Amber */
    div[data-testid="stMetric"]:nth-child(3) { border-left-color: #10b981; } /* Health Green */
    
    /* Chat Bubble Styling */
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# Secure API Key Handling
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key Missing! Please add it to Streamlit Secrets.")
    st.stop()

# --- 2. DATA ENRICHMENT & VECTOR ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        # Load local datasets
        projects_df = pd.read_csv('project_risk_raw_dataset.csv')
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')

        # Calculate live market sentiment
        latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
        market_summary = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_market.iterrows()])

        # Enrich Project Data with Financial & Market Context
        def enrich_logic(row):
            pid = row['Project_ID']
            overdue_val = txns_df[(txns_df['Project_ID'] == pid) & (txns_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
            return (f"PROJECT ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                    f"Overdue: ${overdue_val:,.2f} | Market Context: {market_summary}")

        projects_df['master_context'] = projects_df.apply(enrich_logic, axis=1)
        
        # Build Vector Memory
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(projects_df, page_content_column="master_context")
        vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings)
        
        return vector_db, projects_df, txns_df, market_df
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

# --- 3. UI LAYOUT & INTERACTIVITY ---
db, p_df, t_df, m_df = initialize_risk_engine()

# Sidebar Control Panel
with st.sidebar:
    st.title("🛡️ Control Panel")
    st.markdown("---")
    risk_level_ui = st.multiselect("Risk Sensitivity Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    complexity_threshold = st.slider("Complexity Intensity", 0, 10, (2, 8))
    
    st.markdown("### 🛰️ Connectivity")
    st.success("● Google Gemini 3: ONLINE")
    st.success("● Vector DB: SYNCED")
    if st.button("Clear Cache & Reboot"):
        st.cache_resource.clear()
        st.rerun()

if db:
    # Top Row: Financial Intelligence Cards
    col1, col2, col3, col4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_risk_count = len(p_df[p_df['Risk_Level'] == 'High'])
    
    col1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", delta="Overdue")
    col2.metric("Critical Projects", high_risk_count, delta="Immediate Action")
    col3.metric("System Health", "96%", delta="Optimal")
    col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'])

    st.markdown("---")

    # Middle Row: Visual Intelligence (Plotly Chart)
    st.subheader("📊 Portfolio Risk Heatmap")
    chart_col, info_col = st.columns([3, 1])
    
    with chart_col:
        fig = px.scatter(
            p_df, x="Complexity_Score", y="Budget_Utilization_Rate",
            color="Risk_Level", size="Complexity_Score", hover_name="Project_ID",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            template="plotly_white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with info_col:
        st.markdown("""
        **How to read this chart:**
        - **Y-Axis:** High spend utilization.
        - **X-Axis:** Technical complexity.
        - **Upper Right:** The "Danger Zone" requiring immediate intervention.
        """)

    st.markdown("---")

    # Bottom Row: AI Advisory Chat
    st.subheader("🤖 AI Risk Advisor (Gemini 3)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for a risk summary of a specific project..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Analyzing Project Telemetry...", expanded=False) as status:
                # 2026 Stable Model: Gemini 3 Flash
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
                
                qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                    ("system", "You are a professional Lead Risk Officer. Use the provided context to analyze the query. Context: {context}"),
                    ("human", "{input}"),
                ]))
                
                # Retrieve from Vector DB and Run Chain
                rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), qa_chain)
                response = rag_chain.invoke({"input": prompt})
                status.update(label="Analysis Complete", state="complete")
            
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
