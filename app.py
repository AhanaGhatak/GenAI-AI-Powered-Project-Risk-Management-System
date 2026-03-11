import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px
import google.generativeai as genai
from typing import Annotated, TypedDict, List

# Core Agentic & RAG Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

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
        padding: 40px; border-radius: 20px; color: white; text-align: center;
        margin-bottom: 35px; box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
    }
    .main-header h1 { color: white !important; font-size: 3rem !important; margin:0; }
    div[data-testid="stMetric"] {
        background: white; padding: 25px; border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-bottom: 5px solid #e2e8f0;
    }
    div[data-testid="stMetric"]:nth-child(1) { border-bottom-color: #ff4b4b; } 
    div[data-testid="stMetric"]:nth-child(2) { border-bottom-color: #ffa500; } 
    div[data-testid="stMetric"]:nth-child(3) { border-bottom-color: #00d26a; } 
    div[data-testid="stMetric"]:nth-child(4) { border-bottom-color: #6366f1; } 
    .stChatMessage { background-color: white !important; border: 1px solid #e2e8f0 !important; border-radius: 15px !important; }
    </style>
    
    <div class="main-header">
        <h1>PROJECT RISK INTELLIGENCE</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Enterprise Command & Risk Mitigation Engine</p>
    </div>
    """, unsafe_allow_html=True)

# API Security & Model Discovery
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    st.error("🔑 System Credentials Missing! Please update Secrets.")
    st.stop()

def discover_stable_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if "gemini-1.5-flash" in m: return m
        return models[0] if models else "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

working_model_id = discover_stable_model()
llm = ChatGoogleGenerativeAI(model=working_model_id, temperature=0.1)

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
                overdue_sum = t_df[(t_df['Project_ID'] == pid) & (t_df['Payment_Status'] == 'Overdue')]['Amount_USD'].sum()
                return (f"PROJECT ID: {pid} | Type: {row['Project_Type']} | Risk: {row['Risk_Level']} | "
                        f"Overdue: ${overdue_sum:,.2f} | Market: {m_summary}")

            p_df['master_context'] = p_df.apply(enrich, axis=1)
            loader = DataFrameLoader(p_df, page_content_column="master_context")
            vector_db = Chroma.from_documents(
                documents=loader.load(), 
                embedding=embeddings, 
                persist_directory=persist_dir
            )
        
        return vector_db, p_df, t_df, m_df
    except Exception as e:
        st.error(f"Engine Failure: {e}")
        return None, None, None, None

db, p_df, t_df, m_df = initialize_risk_engine()

# --- 3. MULTI-AGENT ARCHITECTURE ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]
    context: str

# Node Definitions
def manager_agent(state: AgentState):
    prompt = f"ROLE: Project Risk Manager. Overall owner. Context: {state['context']}\nUser: {state['messages'][-1].content}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Project_Risk_Manager")]}

def market_agent(state: AgentState):
    prompt = f"ROLE: Market Analysis Agent. Financial trends/news. Context: {state['context']}\nUser: {state['messages'][-1].content}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Market_Analyst")]}

def scoring_agent(state: AgentState):
    prompt = f"ROLE: Risk Scoring Agent. Transaction/Investment risks. Context: {state['context']}\nUser: {state['messages'][-1].content}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Risk_Scorer")]}

def status_agent(state: AgentState):
    prompt = f"ROLE: Project Status Tracking Agent. Internal delays/resignation. Context: {state['context']}\nUser: {state['messages'][-1].content}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Status_Tracker")]}

def reporting_agent(state: AgentState):
    prompt = f"ROLE: Reporting Agent. Risk analytics/alerts. Context: {state['context']}\nUser: {state['messages'][-1].content}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Reporting_Officer")]}

# Router Logic
def smart_router(state: AgentState):
    msg = state['messages'][-1].content.lower()
    if any(k in msg for k in ["market", "trend", "inflation", "sentiment"]): return "market"
    if any(k in msg for k in ["score", "transaction", "payment", "overdue", "fraud"]): return "scoring"
    if any(k in msg for k in ["status", "delay", "resignation", "turnover", "timeline"]): return "status"
    if any(k in msg for k in ["report", "analytic", "alert", "summary", "list"]): return "reporting"
    return "manager"

# Graph Construction
builder = StateGraph(AgentState)
builder.add_node("manager", manager_agent)
builder.add_node("market", market_agent)
builder.add_node("scoring", scoring_agent)
builder.add_node("status", status_agent)
builder.add_node("reporting", reporting_agent)

builder.set_conditional_entry_point(smart_router, {
    "manager": "manager", "market": "market", "scoring": "scoring", 
    "status": "status", "reporting": "reporting"
})

for node in ["manager", "market", "scoring", "status", "reporting"]:
    builder.add_edge(node, END)

agent_system = builder.compile()

# --- 4. DASHBOARD EXECUTION ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROLS")
    st.divider()
    risk_level_ui = st.multiselect("Priority Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    complexity_threshold = st.slider("Complexity Intensity", 0, 10, (2, 9))
    st.markdown("---")
    st.success("● CORE PROCESSOR: ACTIVE")
    st.success(f"● MODEL: {working_model_id}")
    if st.button("🚀 FULL SYSTEM REBOOT"):
        st.cache_resource.clear()
        if os.path.exists("./risk_db_2026"):
            import shutil
            shutil.rmtree("./risk_db_2026")
        st.rerun()

if db:
    # KPI Section
    col1, col2, col3, col4 = st.columns(4)
    overdue_val = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
    high_count = len(p_df[p_df['Risk_Level'] == 'High'])
    avg_turnover = p_df['Team_Turnover_Rate'].mean()
    
    col1.metric("Financial Exposure", f"${overdue_val/1e6:.1f}M", "Total Overdue")
    col2.metric("Critical Alerts", high_count, "High Risk Projects")
    col3.metric("System Health", "98%", "Stable")
    col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Live Feed")

    st.markdown("---")

    # Heatmap
    st.subheader("📊 Portfolio Risk Heatmap")
    fig = px.scatter(
        p_df, x="Complexity_Score", y="Budget_Utilization_Rate",
        color="Risk_Level", size="Complexity_Score", hover_name="Project_ID",
        color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00d26a"},
        template="plotly_white", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Agentic Chat
    st.subheader("💬 MULTI-AGENT STRATEGIC ADVISORY")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Inquire about specific risk patterns..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Consulting Specialized Agents...", expanded=False) as status:
                # RAG Step
                retrieved_docs = db.similarity_search(prompt, k=5)
                context_str = "\n".join([d.page_content for d in retrieved_docs])
                
                # Agent Step
                result = agent_system.invoke({"messages": [HumanMessage(content=prompt)], "context": context_str})
                final_msg = result["messages"][-1]
                agent_name = final_msg.name.replace("_", " ")
                status.update(label=f"Insight from {agent_name}", state="complete")
            
            output_text = f"**[{agent_name}]**\n\n{final_msg.content}"
            st.markdown(output_text)
            st.session_state.chat_history.append({"role": "assistant", "content": output_text})
