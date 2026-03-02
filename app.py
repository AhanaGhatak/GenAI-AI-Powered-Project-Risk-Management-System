import streamlit as st
import pandas as pd
import os

# 2026 Standard: 'chains' are now in 'langchain_classic'
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Control Tower", layout="wide")

# Best Practice: Using Team API Key via Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ GOOGLE_API_KEY not found in Secrets. Please add your teammate's key.")
    st.stop()

# --- DATA ENGINE ---
@st.cache_resource
def initialize_risk_engine():
    try:
        # Load your group project CSV files
        projects_df = pd.read_csv('project_risk_raw_dataset.csv').head(100)
        txns_df = pd.read_csv('transaction.csv')
        market_df = pd.read_csv('market_trends.csv')

        # 1. Market Sentiment Agent (External Data)
        latest_market = market_df.sort_values('Date').groupby('Indicator').tail(1)
        market_context_str = " | ".join([
            f"{row['Indicator']}: {row['Value']} ({row['Market_Sentiment']})" 
            for _, row in latest_market.iterrows()
        ])

        # 2. Risk Data Enrichment (Internal + Financial)
        def enrich_project_data(row):
            p_id = row['Project_ID']
            p_txns = txns_df[txns_df['Project_ID'] == p_id]
            overdue_amt = p_txns[p_txns['Payment_Status'] == 'Overdue']['Amount_USD'].sum()
            
            return (
                f"PROJECT ID: {p_id}. TYPE: {row['Project_Type']}. PHASE: {row['Project_Phase']}. "
                f"SCORE: {row['Complexity_Score']}/10. STATUS: {row['Risk_Level']}. "
                f"FINANCIALS: Overdue Payments ${overdue_amt:,.2f}. "
                f"MARKET CONTEXT: {market_context_str}."
            )

        projects_df['master_context'] = projects_df.apply(enrich_project_data, axis=1)

        # 3. Vector Brain (Stable 2026 Embedding Model)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        loader = DataFrameLoader(projects_df, page_content_column="master_context")
        docs = loader.load()
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./risk_vectors_2026"
        )
        return vector_db
    except Exception as e:
        st.error(f"Error initializing Risk Engine: {e}")
        return None

# --- CHAT AGENT ---
def get_risk_response(query, vector_db):
    # FIXED: Using 'gemini-3-flash-preview' for text reasoning in 2026
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
    
    system_prompt = (
        "You are a Senior Project Risk Consultant. Use the following project data "
        "and market context to answer the query accurately. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Using 'langchain_classic' logic for RAG
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# --- UI LAYER ---
st.title("🛡️ AI Project Risk Control Tower")
st.markdown("---")

risk_db = initialize_risk_engine()

if risk_db:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Ask about project health or financial risks..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing risk report..."):
                answer = get_risk_response(user_input, risk_db)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
