import streamlit as st
import pandas as pd
import os
import plotly.express as px
import shutil

# 2026 Core RAG & Agent Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# --- 1. PREMIUM UI & BRANDING ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 2px solid #e2e8f0; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2 {
        color: #0f172a !important; font-weight: 600 !important;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        padding: 40px; border-radius: 20px; color: white; text-align: center; margin-bottom: 35px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.1);
    }
    .agent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 4px 12px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 800; border: 1px solid #bfdbfe;
        margin-bottom: 8px; display: inline-block; text-transform: uppercase;
    }
    .stChatMessage { border-radius: 15px !important; border: 1px solid #e2e8f0 !important; }
    </style>
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE</h1>
        <p style="font-size: 1.1rem; opacity: 0.85;">Multi-Agent Strategic Command & Control</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing! Check Streamlit Secrets.")
    st.stop()

# --- 2. DATA & VECTOR ENGINE (QUOTA PROTECTED) ---
@st.cache_resource
def initialize_system():
    persist_dir = "./risk_db_agents_v2"
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
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

# --- 3. MULTI-AGENT ARCHITECTURE ---
AGENTS = {
    "Market Analysis Agent": "Analyze external financial trends. How does Market Sentiment affect viability?",
    "Risk Scoring Agent": "Focus on Overdue amounts and Risk Levels. Prioritize the biggest financial threats.",
    "Project Status Agent": "Track complexity, schedule delays, and internal resource risks.",
    "Reporting Agent": "Synthesize data into professional summaries and executive alerts."
}



def run_agent_workflow(query, vector_db):
    # 1. Retrieval
    docs = vector_db.similarity_search(query, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # 2. Supervisor Routing (The Project Risk Manager)
    manager_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    routing_prompt = f"Query: {query}\nPick one agent: {list(AGENTS.keys())}. Reply with ONLY the name."
    
    # Error-safe invocation
    raw_decision = manager_llm.invoke(routing_prompt)
    selected_agent = raw_decision.content.strip() if hasattr(raw_decision, 'content') else str(raw_decision).strip()
    
    # Clean up selection in case of extra text
    for agent in AGENTS.keys():
        if agent.lower() in selected_agent.lower():
            selected_agent = agent
            break

    # 3. Specialist Execution
    specialist_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    agent_instr = AGENTS.get(selected_agent, "Provide general risk advice.")
    
    final_prompt = f"ROLE: {selected_agent}\nMISSION: {agent_instr}\nCONTEXT: {context}\nQUERY: {query}"
    
    raw_response = specialist_llm.invoke(final_prompt)
    response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
    
    return selected_agent, response_text

# --- 4. DASHBOARD RENDER ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROLS")
    st.write("🛰️ **Active Specialists:**")
    for a in AGENTS.keys(): st.caption(f"• {a}")
    st.divider()
    if st.button("🚀 FULL SYSTEM REBOOT"):
        st.cache_resource.clear()
        if os.path.exists("./risk_db_agents_v2"):
            shutil.rmtree("./risk_db_agents_v2")
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
