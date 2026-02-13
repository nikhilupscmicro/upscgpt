import streamlit as st
import os

# Show what's happening during startup
st.write("🔍 Debug: App starting...")

try:
    import chromadb
    st.write("✅ chromadb imported")
except Exception as e:
    st.error(f"❌ chromadb import failed: {e}")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    st.write("✅ langchain_google_genai imported")
except Exception as e:
    st.error(f"❌ langchain_google_genai import failed: {e}")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.chains import RetrievalQA
    st.write("✅ langchain_community imported")
except Exception as e:
    st.error(f"❌ langchain_community import failed: {e}")

# --- 1. BRANDING & UI ---
st.set_page_config(page_title="UPSCGPT", page_icon="🎓")
st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: white; }
    .stTextInput>div>div>input { background-color: #1E293B; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CHECK SECRETS ---
st.write("🔍 Checking secrets...")
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.write("✅ GOOGLE_API_KEY found")
except Exception as e:
    st.error(f"❌ GOOGLE_API_KEY missing: {e}")

try:
    host = st.secrets["CHROMA_CLOUD_HOST"]
    st.write(f"✅ CHROMA_CLOUD_HOST found: {host}")
except Exception as e:
    st.error(f"❌ CHROMA_CLOUD_HOST missing: {e}")

try:
    token = st.secrets["CHROMA_CLOUD_TOKEN"]
    st.write("✅ CHROMA_CLOUD_TOKEN found")
except Exception as e:
    st.error(f"❌ CHROMA_CLOUD_TOKEN missing: {e}")

# --- 3. CLOUD BACKEND CONNECTION ---
@st.cache_resource
def connect_to_upscgpt_brain():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        host = st.secrets["CHROMA_CLOUD_HOST"]
        token = st.secrets["CHROMA_CLOUD_TOKEN"]
        
        st.write("🔍 Initializing embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
        st.write("✅ Embeddings initialized")
        
        st.write("🔍 Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key, 
            temperature=0.1
        )
        st.write("✅ LLM initialized")
        
        st.write("🔍 Connecting to Chroma Cloud...")
        client = chromadb.HttpClient(
            host=host,
            headers={"Authorization": f"Bearer {token}"}
        )
        st.write("✅ Chroma client connected")
        
        st.write("🔍 Loading vector database...")
        vector_db = Chroma(
            client=client,
            collection_name="upsc_collection",
            embedding_function=embeddings
        )
        st.write("✅ Vector DB loaded")
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        raise e

# --- 4. THE INTERFACE ---
st.title("🎓 UPSCGPT")
st.write("Contextual AI for UPSC Aspirants")

try:
    engine = connect_to_upscgpt_brain()
    st.write("✅ Engine ready!")
    
    if prompt := st.chat_input("Ask a doubt from the 2026 Budget or NCERTs..."):
        st.chat_message("user").write(prompt)
        
        with st.spinner("Searching Cloud Knowledge Base..."):
            response = engine.invoke({"query": prompt})
            
            with st.chat_message("assistant"):
                st.write(response["result"])
                
                with st.expander("📚 Verified Sources"):
                    sources = {
                        os.path.basename(doc.metadata.get('source', 'Document')) 
                        for doc in response["source_documents"]
                    }
                    for s in sources:
                        st.write(f"- {s}")
                        
except Exception as e:
    st.error(f"❌ Fatal Error: {e}")
    import traceback
    st.code(traceback.format_exc())
