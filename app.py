import streamlit as st
import pandas as pd
import os
import time

# Modern LangChain Imports (v0.3+)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION & SECRETS ---
st.set_page_config(page_title="AI Project Risk Control Tower", layout="wide")

# It is best practice to store your key in Streamlit Secrets
# If testing locally, you can use: os.environ["GOOGLE_API_KEY"] = "your_key_here"
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please set the GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- DATA ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        # Load Datasets
        projects_df = pd.read_csv('project_risk_raw_dataset.csv').head(100)
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')

        # 1. Summarize Market Trends (External Data Agent)
        latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
        market_context_str = " | ".join([
            f"{row['Indicator']}: {row['Value']} (Sentiment: {row['Market_Sentiment']})" 
            for _, row in latest_market.iterrows()
        ])

        # 2. Enrichment Function (Internal + Financial Agent)
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

        # 3. Vectorization (The Brain)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(projects_df, page_content_column="master_context")
        docs = loader.load()
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./master_risk_brain"
        )
        return vector_db
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# --- CHAT LOGIC ---
def get_risk_response(query, vector_db):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    # Modern Chat Prompt
    system_prompt = (
        "You are the AI Project Risk Manager. Use the following context to provide "
        "a detailed risk report and mitigation strategies. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the modern chain (replacing deprecated RetrievalQA)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# --- STREAMLIT UI ---
st.title("🛡️ AI Project Risk Control Tower")
st.markdown("---")

# Load and Initialize
risk_db = initialize_risk_engine()

if risk_db:
    # Sidebar Agents
    with st.sidebar:
        st.header("Active AI Agents")
        st.success("🟢 Market Analysis Agent")
        st.success("🟢 Risk Scoring Agent")
        st.success("🟢 Status Tracking Agent")
        st.info("Analyzing internal & external telemetry.")

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about project health or specific risks..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing risk report..."):
                response = get_risk_response(prompt, risk_db)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
