import streamlit as st
import chromadb
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# We update this specific line below:
from langchain.chains import RetrievalQA

# --- 1. BRANDING & UI ---
st.set_page_config(page_title="UPSCGPT", page_icon="🎓")

st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: white; }
    .stTextInput>div>div>input { background-color: #1E293B; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CLOUD BACKEND CONNECTION ---
@st.cache_resource
def connect_to_upscgpt_brain():
    # Streamlit pulls these from the 'Secrets' tab in the dashboard
    api_key = st.secrets["GOOGLE_API_KEY"]
    host = st.secrets["CHROMA_CLOUD_HOST"]
    token = st.secrets["CHROMA_CLOUD_TOKEN"]

    # Initialize Gemini 1.5 Flash
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)

    # Connect to the Remote Chroma Cloud
    client = chromadb.HttpClient(
        host=host,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # Load your specific collection
    vector_db = Chroma(
        client=client,
        collection_name="upsc_collection", # DOUBLE CHECK: Make sure this matches your Chroma Cloud name
        embedding_function=embeddings
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# --- 3. THE INTERFACE ---
st.title("🎓 UPSCGPT")
st.write("Contextual AI for UPSC Aspirants")

try:
    engine = connect_to_upscgpt_brain()
    
    if prompt := st.chat_input("Ask a doubt from the 2026 Budget or NCERTs..."):
        st.chat_message("user").write(prompt)
        
        with st.spinner("Searching Cloud Knowledge Base..."):
            response = engine.invoke({"query": prompt})
            
            with st.chat_message("assistant"):
                st.write(response["result"])
                
                # Citations (Crucial for PMs to show "Grounding")
                with st.expander("📚 Verified Sources"):
                    sources = {os.path.basename(doc.metadata.get('source', 'Document')) for doc in response["source_documents"]}
                    for s in sources:
                        st.write(f"- {s}")

except Exception as e:
    st.error(f"Configuration Error: Please verify your Streamlit Cloud Secrets. Error: {e}")
