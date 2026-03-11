import streamlit as st
import pandas as pd
import os
import plotly.express as px
from typing import Annotated, TypedDict, List

# Core LangChain & LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

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
        <p style="font-size: 1.2rem; opacity: 0.9;">Multi-Agent Strategic Mitigation Engine</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE & API ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 System Credentials Missing!")
    st.stop()

@st.cache_data
def load_datasets():
    # Loading full dataset from your uploaded files 
    p_df = pd.read_csv('project_risk_raw_dataset.csv')
    t_df = pd.read_csv('transaction.csv')
    m_df = pd.read_csv('market_trends.csv')
    return p_df, t_df, m_df

p_df, t_df, m_df = load_datasets()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# --- 3. MULTI-AGENT DEFINITIONS ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]

# Agent 1: Project Risk Manager (Orchestrator)
def manager_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"PROJECT DATA OVERVIEW:\n{p_df.describe().to_string()}\n\nROLE: Overall Risk Owner. Identify strategic mitigations for: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Project_Risk_Manager")]}

# Agent 2: Market Analysis Agent
def market_agent(state: AgentState):
    query = state['messages'][-1].content
    recent_m = m_df.tail(20).to_string() # [cite: 25]
    prompt = f"MARKET DATA:\n{recent_m}\n\nROLE: Financial Trend Analyst. Analyze market sentiment and news impacts for: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Market_Analyst")]}

# Agent 3: Risk Scoring Agent
def scoring_agent(state: AgentState):
    query = state['messages'][-1].content
    txn_sample = t_df.head(20).to_string() # [cite: 1]
    prompt = f"TRANSACTION DATA:\n{txn_sample}\n\nROLE: Risk Scorer. Assess transaction risks and payment defaults: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Risk_Scorer")]}

# Agent 4: Project Status Tracking Agent
def status_agent(state: AgentState):
    query = state['messages'][-1].content
    internal_data = p_df[['Project_ID', 'Project_Phase', 'Team_Turnover_Rate', 'Schedule_Pressure']].head(20).to_string() # 
    prompt = f"INTERNAL METRICS:\n{internal_data}\n\nROLE: Status Tracker. Report on resource resignation, delays, and progress: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Status_Tracker")]}

# Agent 5: Reporting Agent
def reporting_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"ROLE: Reporting Specialist. Provide a structured risk executive summary and critical alerts for: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Reporting_Officer")]}

# --- 4. SYSTEM ORCHESTRATION (ROUTING) ---
def router(state: AgentState):
    msg = state['messages'][-1].content.lower()
    if any(k in msg for k in ["market", "inflation", "trend", "sentiment"]): return "market"
    if any(k in msg for k in ["transaction", "payment", "overdue", "default", "score"]): return "scoring"
    if any(k in msg for k in ["status", "delay", "resignation", "turnover", "progress"]): return "status"
    if any(k in msg for k in ["report", "summary", "analytics", "alert"]): return "reporting"
    return "manager"

builder = StateGraph(AgentState)
builder.add_node("manager", manager_agent)
builder.add_node("market", market_agent)
builder.add_node("scoring", scoring_agent)
builder.add_node("status", status_agent)
builder.add_node("reporting", reporting_agent)

builder.set_conditional_entry_point(router, {
    "manager": "manager", "market": "market", "scoring": "scoring", 
    "status": "status", "reporting": "reporting"
})

for node in ["manager", "market", "scoring", "status", "reporting"]:
    builder.add_edge(node, END)

agent_system = builder.compile()

# --- 5. DASHBOARD & SIDEBAR ---
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROLS")
    st.divider()
    risk_level_ui = st.multiselect("Priority Focus", ["High", "Medium", "Low"], default=["High", "Medium"])
    st.success("● 5 AGENTS ONLINE")
    st.info(f"Loaded {len(p_df)} Projects")
    if st.button("🚀 FULL SYSTEM REBOOT"):
        st.cache_data.clear()
        st.rerun()

# KPI SECTION 
col1, col2, col3, col4 = st.columns(4)
overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
high_risk_count = len(p_df[p_df['Risk_Level'] == 'High'])
avg_turnover = p_df['Team_Turnover_Rate'].mean() * 100

col1.metric("Financial Exposure", f"${overdue_total/1e6:.1f}M", "Total Overdue")
col2.metric("Critical Alerts", high_risk_count, "High Priority")
col3.metric("Avg. Turnover", f"{avg_turnover:.1f}%", "Internal Risk")
col4.metric("Market Sentiment", m_df.iloc[-1]['Market_Sentiment'], "Current Trend")

st.markdown("---")

# HEATMAP SECTION
st.subheader("📊 Portfolio Risk Heatmap")
fig = px.scatter(
    p_df[p_df['Risk_Level'].isin(risk_level_ui)], 
    x="Complexity_Score", y="Budget_Utilization_Rate",
    color="Risk_Level", size="Team_Size", hover_name="Project_ID",
    color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00d26a"},
    template="plotly_white", height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- 6. AGENTIC ADVISORY CHAT ---
st.subheader("💬 MULTI-AGENT STRATEGIC ADVISORY")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an agent: e.g., 'Report on high-turnover projects' or 'Analyze market risks'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Orchestrating Specialist Agents...", expanded=False) as status:
            result = agent_system.invoke({"messages": [HumanMessage(content=prompt)]})
            ans_msg = result["messages"][-1]
            agent_name = ans_msg.name.replace("_", " ")
            status.update(label=f"Response from {agent_name}", state="complete")
        
        full_response = f"**{agent_name}**: {ans_msg.content}"
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
