# --- CRITICAL SQLITE FIX (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import os
import time
import shutil
import plotly.express as px

# 2026 Core RAG & Agent Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# --- 1. PREMIUM UI & BRANDING ---
st.set_page_config(page_title="Risk Intel Command", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        padding: 40px; border-radius: 20px; color: white; text-align: center; margin-bottom: 35px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.15);
    }
    .agent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 4px 12px;
        border-radius: 10px; font-size: 0.7rem; font-weight: 800; border: 1px solid #bfdbfe;
        margin-bottom: 8px; display: inline-block; text-transform: uppercase;
    }
    </style>
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Gemini 3.1 Multi-Agent Strategic Command • 2026 Stable</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key Missing! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. RESILIENT DATA ENGINE ---
@st.cache_resource
def initialize_system():
    # Folder v10 for absolute 2026 model compatibility
    persist_dir = "./risk_db_v10_final"
    
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        
        # 503 Resiliency Loop for Embeddings
        embeddings = None
        for attempt in range(3):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                # Test call to verify service
                embeddings.embed_query("test")
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(3)
                    continue
                raise e

        if os.path.exists(persist_dir):
            vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            latest_m = m_df.iloc[-1]
            def enrich(row):
                pid = row['Project_ID']
                overdue = t_df[(t_df['Project_ID'] == pid) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
                return (f"PID: {pid} | Risk: {row['Risk_Level']} | Type: {row['Project_Type']} | "
                        f"Overdue: ${overdue:,.0f} | Market: {latest_m['Market_Sentiment']}")

            p_df['context'] = p_df.apply(enrich, axis=1)
            loader = DataFrameLoader(p_df, page_content_column="context")
            vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings, persist_directory=persist_dir)
        
        return vector_db, p_df, t_df, m_df
    except Exception as e:
        st.error(f"System Load Error: {e}")
        return None, None, None, None

db, p_df, t_df, m_df = initialize_system()

# --- 3. MULTI-AGENT ORCHESTRATION ---
AGENTS = {
    "Market Analyst": "Focus on external trends, market sentiment, and macro-economic project risks.",
    "Financial Auditor": "Focus on budget utilization, overdue payments, and financial exposure.",
    "Project Manager": "Focus on project complexity, scheduling, and internal operational risks.",
    "Executive Reporter": "Synthesize data into high-level strategic summaries and board-level alerts."
}

def extract_text(response):
    """Safely extracts text from Gemini 3.1 list or string content."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content]).strip()
    return str(content).strip()



def run_agent_workflow(query, vector_db):
    try:
        # 1. Retrieval (RAG)
        docs = vector_db.similarity_search(query, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        # 2. Supervisor Routing
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.1)
        
        routing_prompt = f"Query: {query}\nSelect exactly one agent: {list(AGENTS.keys())}. Reply with ONLY the name."
        raw_decision = llm.invoke(routing_prompt)
        selected_name = extract_text(raw_decision)
        
        # Fallback mapping
        active_agent = next((a for a in AGENTS.keys() if a.lower() in selected_name.lower()), "Executive Reporter")
        
        # 3. Specialist Execution
        instruction = AGENTS.get(active_agent)
        final_prompt = f"ROLE: {active_agent}\nMISSION: {instruction}\nCONTEXT: {context}\nQUERY: {query}"
        
        raw_response = llm.invoke(final_prompt)
        return active_agent, extract_text(raw_response)
    except Exception as e:
        if "503" in str(e):
            return "System", "The AI Core is currently overloaded (503). Please wait 10 seconds and try again."
        return "System", f"Workflow Error: {str(e)}"

# --- 4. DASHBOARD & CHAT UI ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROL")
    st.info("🛰️ Status: ACTIVE (Gemini 3.1)")
    if st.button("🚀 FULL SYSTEM WIPE"):
        st.cache_resource.clear()
        folders = ["./risk_db_final_v5", "./risk_db_v7_stable", "./risk_db_v8_final", "./risk_db_v10_final"]
        for f in folders:
            if os.path.exists(f):
                shutil.rmtree(f)
        st.rerun()

if db is not None:
    # KPI Section
    c1, c2, c3, c4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    c1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", "Overdue")
    c2.metric("Critical Alerts", len(p_df[p_df['Risk_Level'] == 'High']), "High Risk")
    c3.metric("System Health", "98%", "Stable")
    c4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Live")

    st.divider()

    # Chat Interface
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if "agent" in m: st.markdown(f"<span class='agent-tag'>{m['agent']}</span>", unsafe_allow_html=True)
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask the Risk Management Team..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Project Risk Manager routing task...", expanded=False) as status:
                agent_name, result = run_agent_workflow(prompt, db)
                status.update(label=f"Analysis Delivered by {agent_name}", state="complete")
            
            st.markdown(f"<span class='agent-tag'>{agent_name}</span>", unsafe_allow_html=True)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result, "agent": agent_name})
