import streamlit as st
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Manager", layout="wide")
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY" # Recommendation: Use st.secrets
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- DATA PROCESSING & VECTOR DB SETUP ---
@st.cache_resource
def initialize_risk_engine():
    # 1. Load Datasets (Assuming files exist in directory)
    try:
        projects_df = pd.read_csv('project_risk_raw_dataset.csv').head(100)
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None

    # 2. Summarize Market Trends
    latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
    market_context_str = " | ".join([
        f"{row['Indicator']}: {row['Value']} (Sentiment: {row['Market_Sentiment']})" 
        for _, row in latest_market.iterrows()
    ])

    # 3. Enrichment Function
    def enrich_project_data(row):
        p_id = row['Project_ID']
        p_txns = txns_df[txns_df['Project_ID'] == p_id]
        total_invoiced = p_txns['Amount_USD'].sum()
        overdue_txns = p_txns[p_txns['Payment_Status'] == 'Overdue']
        overdue_amt = overdue_txns['Amount_USD'].sum()
        
        return (
            f"PROJECT PROFILE: {p_id} ({row['Project_Type']}). "
            f"PHASE: {row['Project_Phase']}. RISK LEVEL: {row['Risk_Level']}. "
            f"INTERNAL HEALTH: Complexity {row['Complexity_Score']}/10, "
            f"Schedule Pressure {row['Schedule_Pressure']}, "
            f"Budget Util {row['Budget_Utilization_Rate']}. "
            f"FINANCIAL STATUS: Total Invoiced ${total_invoiced:,.2f}. "
            f"Overdue Amount: ${overdue_amt:,.2f} ({len(overdue_txns)} invoices). "
            f"EXTERNAL MARKET CONTEXT: {market_context_str}."
        )

    projects_df['master_context'] = projects_df.apply(enrich_project_data, axis=1)

    # 4. Vectorize to ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    loader = DataFrameLoader(projects_df, page_content_column="master_context")
    docs = loader.load()
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./master_risk_brain"
    )
    return vector_db

# --- AGENT LOGIC ---
def get_risk_response(query, vector_db):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    template = """
    You are the AI Project Risk Manager. You have access to detailed project health, financial transactions, and market trends.
    Your goal is to act as the Project Risk Manager, Market Analyst, and Reporting Agent combined.
    
    Use the following project context to answer the user's question. 
    If you don't know the answer, just say you don't have that specific data.
    
    Context: {context}
    Question: {question}
    
    Detailed Risk Analysis:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain.invoke({"query": query})["result"]

# --- STREAMLIT UI ---
st.title("🛡️ AI Project Risk Control Tower")
st.markdown("---")

# Initialize DB
with st.spinner("Analyzing risk datasets and syncing Market Trends..."):
    risk_db = initialize_risk_engine()

if risk_db:
    # Sidebar for project overview
    st.sidebar.header("Agent Status")
    st.sidebar.success("✅ Market Analysis Agent: Online")
    st.sidebar.success("✅ Risk Scoring Agent: Active")
    st.sidebar.info("System monitoring 100+ project parameters.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about project risks (e.g., 'Which projects have high overdue amounts?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_risk_response(prompt, risk_db)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
