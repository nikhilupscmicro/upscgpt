import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. SETUP & THEME
st.set_page_config(page_title="UPSCgtp | AI", page_icon="🎓", layout="centered")

# Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #0F172A; }
    .stTextInput>div>div>input { background-color: #1E293B; color: white; border-radius: 10px; }
    .stMarkdown { color: #F8FAFC; }
    .answer-box { background-color: #1E293B; padding: 20px; border-radius: 15px; border-left: 5px solid #38BDF8; }
    </style>
""", unsafe_allow_html=True)

# 2. THE BACKEND (AI Connection)
# This function runs once and stays in the "memory" of the app
@st.cache_resource
def initialize_upsc_brain():
    # Make sure your API Key is set in your VPS environment or replace 'YOUR_KEY' here
    api_key = "YOUR_GEMINI_API_KEY_HERE" 
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    # This points to the folder you uploaded to your VPS
    vector_db = Chroma(persist_directory="./upsc_vector_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
    
    # Create the 'Question-Answer' Engine
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# 3. THE FRONTEND (The App Interface)
st.title("🎓 UPSCgtp")
st.subheader("Your AI-Powered Content Repository")

# Initialize the engine
with st.spinner("Initializing AI Brain..."):
    upsc_engine = initialize_upsc_brain()

# The Search Box
query = st.text_input("Ask a question (e.g., 'What is Fiscal Deficit?'):")

if query:
    with st.spinner("Analyzing documents..."):
        # Run the backend logic
        response = upsc_engine.invoke({"query": query})
        
        # Display the result
        st.markdown("### 📝 AI Synthesis")
        st.markdown(f'<div class="answer-box">{response["result"]}</div>', unsafe_allow_html=True)
        
        # Display the "Sources" (The PM move for trust)
        with st.expander("📚 Sources Cited"):
            for doc in response["source_documents"]:
                st.write(f"- {os.path.basename(doc.metadata['source'])} (Page {doc.metadata.get('page', 'N/A')})")

st.info("Built for UPSC Aspirants. Grounded in Official Government Reports.")