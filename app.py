import streamlit as st
import pandas as pd
import os

# 2026 Modular LangChain Imports
# Chains and older RAG patterns are now in 'langchain_classic'
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Control Tower", layout="wide")

# Best Practice: Use Streamlit Secrets for the Team API Key
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to Streamlit Secrets.")
    st.stop()

# --- DATA ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        # Load Datasets (Ensure these files are in your GitHub repo)
        projects_df = pd.read_csv('project_risk_raw_dataset.csv').head(100)
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')

        # 1. External Market Context Agent
        latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
        market_context_str = " | ".join([
            f"{row['Indicator']}: {row['Value']} ({row['Market_Sentiment']})" 
            for _, row in latest_market.iterrows()
        ])

        # 2. Risk Data Enrichment
        def enrich_project_data(row):
            p_id = row['Project_ID']
            p_txns = txns_df[txns_df['Project_ID'] == p_id]
            overdue_amt = p_txns[p_txns['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
            
            return (
                f"PROJECT: {p_id} ({row['Project_Type']}). PHASE: {row['Project_Phase']}. "
                f"RISK: {row['Risk_Level']}. BUDGET UTIL: {row['Budget_Utilization_Rate']}. "
                f"FINANCIALS: Overdue ${overdue_amt:,.2f}. "
                f"MARKET TRENDS: {market_context_str}."
            )

        projects_df['master_context'] = projects_df.apply(enrich_project_data, axis=1)

        # 3. Vector Brain (Updated to 2026 stable embedding model)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(projects_df, page_content_column="master_context")
        docs = loader.load()
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./risk_db_2026"
        )
        return vector_db
    except Exception as e:
        st.error(f"Engine Initialization Failed: {e}")
        return None

# --- RISK AGENT LOGIC ---
def get_risk_response(query, vector_db):
    # Updated to Gemini 3.1 Flash (The March 2026 Standard)
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-preview", temperature=0.1)
    
    system_prompt = (
        "You are the Lead AI Risk Officer. Use the provided context to analyze "
        "project health and recommend mitigation. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Using the modern chain from langchain_classic
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# --- STREAMLIT UI ---
st.title("🛡️ AI Project Risk Control Tower")
st.markdown("---")

risk_db = initialize_risk_engine()

if risk_db:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask about project risks..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing risk telemetry..."):
                answer = get_risk_response(user_input, risk_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
