import streamlit as st
import pandas as pd
import os
import plotly.express as px

# 2026 Core RAG & Agent Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# --- 1. PREMIUM UI & BRANDING STYLING ---
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
    }
    .agent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 4px 10px;
        border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid #bfdbfe;
        margin-bottom: 10px; display: inline-block;
    }
    </style>
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Multi-Agent Collaborative Command Center</p>
    </div>
    """, unsafe_allow_html=True)

# API Security
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing! Please update Secrets.")
    st.stop()

# --- 2. PERSISTENT DATA ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    persist_dir = "./risk_db_2026"
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        if os.path.exists(persist_dir):
            vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            latest_market = m_df.sort_values('Date').groupby('Indicator').tail(1)
            m_summary = " | ".join([f"{r['Indicator']}: {r['Value']}" for _, r in latest_market.iterrows()])

            def enrich(row):
                pid = row['Project_ID']
                overdue = t_df[(t_df['Project_ID'] == pid) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
                return (f"PROJECT ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                        f"Overdue: ${overdue:,.2f} | Market: {m_summary}")

            p_df['master_context'] = p_df.apply(enrich, axis=1)
            loader = DataFrameLoader(p_df, page_content_column="master_context")
            vector_db = Chroma.from_documents(documents=loader.load(), embedding=embeddings, persist_directory=persist_dir)
        
        return vector_db, p_df, t_df, m_df
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

db, p_df, t_df, m_df = initialize_risk_engine()

# --- 3. MULTI-AGENT ORCHESTRATION ---
# 

AGENTS = {
    "Market Analysis Agent": "Analyze financial trends and news. Focus on how Market Sentiment and economic indicators affect the project portfolio.",
    "Risk Scoring Agent": "Assess transaction and investment risks. Focus strictly on Overdue amounts and Risk Levels to prioritize financial danger.",
    "Project Status Agent": "Track project progress and internal risks like complexity, schedule delays, or resource resignations.",
    "Reporting Agent": "Synthesize data into detailed risk analytics, structured alerts, and executive summaries."
}

def run_agent_workflow(query, vector_db):
    # 1. Retrieval (RAG)
    docs = vector_db.similarity_search(query, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # 2. Project Risk Manager (The Supervisor/Router)
    manager_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    routing_prompt = f"User Question: {query}\nWhich agent should handle this: {list(AGENTS.keys())}? Reply with ONLY the name."
    selected_agent = manager_llm.invoke(routing_prompt).content.strip()
    
    # 3. Specialized Agent Execution
    specialist_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
    agent_instructions = AGENTS.get(selected_agent, "Provide general risk advice.")
    
    final_prompt = f"""
    ROLE: {selected_agent}
    MISSION: {agent_instructions}
    CONTEXT FROM DATABASE: {context}
    USER QUERY: {query}
    
    Provide a professional analysis based on the context above.
    """
    response = specialist_llm.invoke(final_prompt)
    return selected_agent, response.content

# --- 4. DASHBOARD UI ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROLS")
    st.divider()
    st.write("🛰️ **Active Agents:**")
    for agent in AGENTS.keys():
        st.caption(f"• {agent}")
    
    st.divider()
    if st.button("🚀 FULL SYSTEM REBOOT"):
        st.cache_resource.clear()
        if os.path.exists("./risk_db_2026"):
            import shutil
            shutil.rmtree("./risk_db_2026")
        st.rerun()

if db:
    # KPI SECTION
    col1, col2, col3, col4 = st.columns(4)
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    col1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", "Total Overdue")
    col2.metric("Critical Alerts", len(p_df[p_df['Risk_Level'] == 'High']), "High Priority")
    col3.metric("System Health", "98%", "Stable")
    col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Live Feed")

    st.markdown("---")
    
    # VISUALIZATION
    st.subheader("📊 Portfolio Risk Heatmap")
    fig = px.scatter(p_df, x="Complexity_Score", y="Budget_Utilization_Rate", color="Risk_Level", 
                     size="Complexity_Score", hover_name="Project_ID", 
                     color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00d26a"},
                     template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # MULTI-AGENT ADVISOR SECTION
    st.subheader("💬 MULTI-AGENT STRATEGIC CHAT")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "agent" in msg:
                st.markdown(f"<span class='agent-tag'>{msg['agent']}</span>", unsafe_allow_html=True)
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the Risk Management Team..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Project Risk Manager routing task...", expanded=False) as status:
                agent_name, result = run_agent_workflow(prompt, db)
                status.update(label=f"Response from {agent_name}", state="complete")
            
            st.markdown(f"<span class='agent-tag'>{agent_name}</span>", unsafe_allow_html=True)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result, "agent": agent_name})
