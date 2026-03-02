__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import shutil

# 2026 Standard Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# --- 1. UI SETUP ---
st.set_page_config(page_title="Risk Intel Multi-Agent", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        padding: 40px; border-radius: 20px; color: white; text-align: center; margin-bottom: 35px;
    }
    .agent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 4px 12px;
        border-radius: 10px; font-size: 0.7rem; font-weight: 800; border: 1px solid #bfdbfe;
        margin-bottom: 8px; display: inline-block;
    }
    </style>
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE</h1>
        <p>Gemini 3.1 Stable Multi-Agent Architecture • 2026</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 API Key Missing! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. DATA ENGINE (FIXED MODELS) ---
@st.cache_resource
def initialize_system():
    # Folder v8 ensures we avoid any previous version conflicts
    persist_dir = "./risk_db_v8_final"
    
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        
        # 2026 Stable Embedding Model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
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

# --- 3. AGENT LOGIC (FIXED FOR LIST/STRIP ERROR) ---
AGENTS = {
    "Market Analyst": "Evaluate external sentiment and financial trends.",
    "Financial Auditor": "Analyze overdue payments and budget utilization.",
    "Project Manager": "Assess project complexity and internal delays.",
    "Executive Reporter": "Summarize all risks for high-level stakeholders."
}

def extract_text(response):
    """Safely extracts text from Gemini 3 response objects/lists."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content]).strip()
    return str(content).strip()



def run_agent_workflow(query, vector_db):
    try:
        # RAG Search
        docs = vector_db.similarity_search(query, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        # 2026 Stable Flash Model
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.1)
        
        # 1. Routing
        routing_prompt = f"Query: {query}\nSelect one agent: {list(AGENTS.keys())}. Reply with the NAME ONLY."
        raw_decision = llm.invoke(routing_prompt)
        selected_name = extract_text(raw_decision)
        
        active_agent = next((a for a in AGENTS.keys() if a.lower() in selected_name.lower()), "Executive Reporter")
        
        # 2. Specialist Response
        instruction = AGENTS.get(active_agent)
        final_prompt = f"ROLE: {active_agent}\nGOAL: {instruction}\nDATA: {context}\nUSER: {query}"
        
        raw_response = llm.invoke(final_prompt)
        return active_agent, extract_text(raw_response)
    except Exception as e:
        return "System", f"Agent Workflow Error: {str(e)}"

# --- 4. DASHBOARD ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROL")
    if st.button("🚀 FULL SYSTEM WIPE"):
        st.cache_resource.clear()
        # Clean up all legacy folders
        folders = ["./risk_db_agents_v3", "./risk_db_final_v5", "./risk_db_v7_stable", "./risk_db_v8_final"]
        for f in folders:
            if os.path.exists(f):
                shutil.rmtree(f)
        st.rerun()

if db:
    # Dashboard Metrics
    cols = st.columns(4)
    total_overdue = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    cols[0].metric("Financial Exposure", f"${total_overdue/1e6:.1f}M")
    cols[1].metric("Critical Risks", len(p_df[p_df['Risk_Level'] == 'High']))
    cols[2].metric("Market Index", m_df.iloc[-1]['Market_Sentiment'])
    cols[3].metric("Active Agents", len(AGENTS))

    st.divider()

    # Chat Interface
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if "agent" in m: st.markdown(f"<span class='agent-tag'>{m['agent']}</span>", unsafe_allow_html=True)
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask the Risk Advisory Team..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Consulting Specialist Agents...", expanded=False) as s:
                agent_name, result = run_agent_workflow(prompt, db)
                s.update(label=f"Analysis Complete by {agent_name}", state="complete")
            
            st.markdown(f"<span class='agent-tag'>{agent_name}</span>", unsafe_allow_html=True)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result, "agent": agent_name})
