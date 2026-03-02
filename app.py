import streamlit as st
import pandas as pd
import os
import plotly.express as px
import shutil

# 2026 Core RAG & Agent Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# --- 1. UI & BRANDING ---
st.set_page_config(page_title="Risk Intel Multi-Agent", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 2px solid #e2e8f0; }
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
        <p>Multi-Agent Collaborative Command Center • v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing! Check Streamlit Secrets.")
    st.stop()

# --- 2. DATA & VECTOR ENGINE ---
@st.cache_resource
def initialize_system():
    persist_dir = "./risk_db_agents_v3"
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        
        # Standard Stable Embedding Model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if os.path.exists(persist_dir):
            vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            latest_m = m_df.iloc[-1]
            def enrich(row):
                pid = row['Project_ID']
                overdue = t_df[(t_df['Project_ID'] == pid) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
                return (f"ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                        f"Overdue: ${overdue:,.2f} | Market Sentiment: {latest_m['Market_Sentiment']}")

            p_df['context'] = p_df.apply(enrich, axis=1)
            loader = DataFrameLoader(p_df, page_content_column="context")
            vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings, persist_directory=persist_dir)
        
        return vector_db, p_df, t_df, m_df
    except Exception as e:
        st.error(f"System Load Error: {e}")
        return None, None, None, None

db, p_df, t_df, m_df = initialize_system()

# --- 3. MULTI-AGENT DEFINITIONS ---
AGENTS = {
    "Market Analysis Agent": "Analyze financial trends and news. Focus on how Market Sentiment affects viability.",
    "Risk Scoring Agent": "Focus strictly on Overdue amounts and Risk Levels to prioritize financial danger.",
    "Project Status Agent": "Track complexity, schedule delays, and internal resource risks.",
    "Reporting Agent": "Synthesize data into professional summaries and executive alerts."
}



def run_agent_workflow(query, vector_db):
    # 1. Retrieval (RAG)
    docs = vector_db.similarity_search(query, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # 2. Supervisor Routing
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    routing_prompt = f"Given this query: '{query}', which agent should handle it? {list(AGENTS.keys())}. Respond with ONLY the agent name."
    
    try:
        raw_decision = llm.invoke(routing_prompt)
        selected_agent = raw_decision.content.strip()
        
        # Clean up agent name mapping
        matched_agent = "Reporting Agent" # Default fallback
        for agent in AGENTS.keys():
            if agent.lower() in selected_agent.lower():
                matched_agent = agent
                break
    except:
        matched_agent = "Reporting Agent"

    # 3. Specialist Execution
    agent_instr = AGENTS.get(matched_agent)
    final_prompt = f"ROLE: {matched_agent}\nMISSION: {agent_instr}\nCONTEXT: {context}\nQUERY: {query}"
    
    try:
        raw_response = llm.invoke(final_prompt)
        return matched_agent, raw_response.content
    except Exception as e:
        return "System Error", f"The AI encountered a safety or quota block: {str(e)}"

# --- 4. DASHBOARD RENDER ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROLS")
    st.write("🛰️ **Active Agents:**")
    for a in AGENTS.keys(): st.caption(f"• {a}")
    st.divider()
    if st.button("🚀 FULL SYSTEM REBOOT"):
        st.cache_resource.clear()
        if os.path.exists("./risk_db_agents_v3"):
            shutil.rmtree("./risk_db_agents_v3")
        st.rerun()

if db is not None:
    # KPI SECTION
    c1, c2, c3, c4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    c1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", "Total Overdue")
    c2.metric("Critical Alerts", len(p_df[p_df['Risk_Level'] == 'High']), "High Risk")
    c3.metric("System Status", "Healthy", "Operational")
    c4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Live Index")

    st.divider()
    
    # HEATMAP
    st.subheader("📊 Portfolio Risk Heatmap")
    fig = px.scatter(p_df, x="Complexity_Score", y="Budget_Utilization_Rate", color="Risk_Level",
                     size="Complexity_Score", hover_name="Project_ID",
                     color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                     template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # CHAT INTERFACE
    st.subheader("💬 STRATEGIC ADVISORY CHAT")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "agent" in msg: st.markdown(f"<span class='agent-tag'>{msg['agent']}</span>", unsafe_allow_html=True)
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query the Risk Management Team..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Project Risk Manager routing task...", expanded=False) as status:
                agent_name, result = run_agent_workflow(prompt, db)
                status.update(label=f"Response from {agent_name}", state="complete")
            
            st.markdown(f"<span class='agent-tag'>{agent_name}</span>", unsafe_allow_html=True)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result, "agent": agent_name})
