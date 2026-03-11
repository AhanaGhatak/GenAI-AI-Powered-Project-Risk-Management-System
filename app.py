import os
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from typing import Annotated, TypedDict, List

# Core Agentic Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. LIGHT-THEME UI SETUP ---
st.set_page_config(page_title="Risk Intel Pro - Suite", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .main-header {
        background: #1e3a8a; /* Deep Blue */
        padding: 25px; border-radius: 12px; text-align: center;
        margin-bottom: 30px; border-bottom: 5px solid #2563eb;
    }
    .main-header h1 { color: #ffffff !important; font-size: 2.2rem; margin: 0; }
    .main-header p { color: #bfdbfe; font-size: 1rem; }
    .metric-container {
        background: #f8fafc; border: 1px solid #e2e8f0;
        padding: 15px; border-radius: 10px; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label { color: #64748b; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
    .metric-value { color: #0f172a; font-size: 1.6rem; font-weight: 800; }
    .stChatMessage { background-color: #f1f5f9 !important; border-radius: 10px; }
    </style>
    
    <div class="main-header">
        <h1>🛡️ ENTERPRISE RISK COMMAND CENTER</h1>
        <p>Multi-Agent Intelligence for Projects, Transactions & Markets</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. AUTH & MODEL ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    st.error("🔑 API Key Missing!")
    st.stop()

@st.cache_resource
def discover_model():
    return "gemini-1.5-flash"

working_model_id = discover_model()
llm = ChatGoogleGenerativeAI(model=working_model_id, temperature=0)

# --- 3. DATA ENGINE ---
@st.cache_data
def load_all_data():
    try:
        p_df = pd.read_csv('project_risk_raw_dataset.csv')
        t_df = pd.read_csv('transaction.csv')
        m_df = pd.read_csv('market_trends.csv')
        return p_df, t_df, m_df
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

p_df, t_df, m_df = load_all_data()

def get_safe_col(df, options):
    for opt in options:
        if opt in df.columns: return opt
    return df.columns[0] if not df.empty else None

# --- 4. THE AGENTIC SUITE ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]

# 1. Project Risk Manager (Orchestrator/Strategy)
def manager_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"PROJECT DATA:\n{p_df.head(10).to_string()}\n\nROLE: Overall Risk Owner. Identify strategic mitigations for: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Risk_Manager")]}

# 2. Market Analysis Agent (Trends)
def market_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"MARKET DATA:\n{m_df.head(15).to_string()}\n\nROLE: Financial Trend Analyst. Analyze external news and trends: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Market_Analyst")]}

# 3. Risk Scoring Agent (Transactions)
def scoring_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"TRANSACTION DATA:\n{t_df.head(20).to_string()}\n\nROLE: Financial Risk Scorer. Assess transaction and investment risks: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Risk_Scorer")]}

# 4. Project Status Tracking Agent (Internal)
def status_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"PROJECT DATA:\n{p_df.head(15).to_string()}\n\nROLE: Status Tracker. Report on delays, resignations, and timeline slips: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Status_Tracker")]}

# 5. Reporting Agent (Analytics & Alerts)
def reporting_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"Provide a detailed risk analytic summary and set of high-priority alerts for: {query}"
    res = llm.invoke(prompt)
    return {"messages": [AIMessage(content=res.content, name="Reporting_Officer")]}

# --- SMART ROUTER ---
def router(state: AgentState):
    msg = state['messages'][-1].content.lower()
    if any(k in msg for k in ["market", "trend", "inflation", "economy"]): return "market"
    if any(k in msg for k in ["transaction", "payment", "fraud", "score", "invest"]): return "scoring"
    if any(k in msg for k in ["delay", "resignation", "timeline", "status", "progress"]): return "status"
    if any(k in msg for k in ["report", "alert", "analytic", "summary"]): return "reporting"
    return "manager"

# Build Graph
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

agent_brain = builder.compile()

# --- 5. DASHBOARD DISPLAY ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-container" style="border-top: 4px solid #ef4444;"><div class="metric-label">Critical Risks</div><div class="metric-value">12</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-container" style="border-top: 4px solid #f59e0b;"><div class="metric-label">Market Sent.</div><div class="metric-value">0.42</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-container" style="border-top: 4px solid #3b82f6;"><div class="metric-label">Overdue TXNs</div><div class="metric-value">8</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-container" style="border-top: 4px solid #8b5cf6;"><div class="metric-label">Staff Turnover</div><div class="metric-value">14%</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-container" style="border-top: 4px solid #10b981;"><div class="metric-label">System Health</div><div class="metric-value">OPTIMAL</div></div>', unsafe_allow_html=True)

st.write("")
col_l, col_r = st.columns([2, 1])

with col_l:
    if not p_df.empty:
        fig = px.bar(p_df.head(20), x='Project_ID', y='Complexity_Score', color='Risk_Level', 
                     title="Project Risk vs Complexity", barmode='group',
                     color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'})
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    if not t_df.empty:
        fig2 = px.pie(t_df.head(50), names='Payment_Status', title="Transaction Exposure", hole=0.5,
                      color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b'])
        st.plotly_chart(fig2, use_container_width=True)

# --- 6. CHAT INTERFACE ---
st.markdown("<h3 style='color: #1e3a8a;'>💬 Enterprise Intelligence Chat</h3>", unsafe_allow_html=True)
if "history" not in st.session_state: st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Ask any agent (e.g., 'Report all delayed projects' or 'Analyze market trends')..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.spinner("🤖 Orchestrating agents..."):
        try:
            result = agent_brain.invoke({"messages": [HumanMessage(content=prompt)]})
            ans = result["messages"][-1]
            full_res = f"**{ans.name.replace('_', ' ')}**: {ans.content}"
            st.chat_message("assistant").write(full_res)
            st.session_state.history.append({"role": "assistant", "content": full_res})
        except Exception as e:
            st.error(f"Error: {e}")
