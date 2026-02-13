import streamlit as st
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. UI SETUP
st.set_page_config(page_title="UPSCgtp | Cloud AI", layout="centered")

# 2. BACKEND (Cloud Connection)
@st.cache_resource
def init_upsc_cloud():
    # These secrets will be set in Streamlit Cloud "Advanced Settings"
    api_key = st.secrets["GOOGLE_API_KEY"]
    chroma_host = st.secrets["CHROMA_CLOUD_HOST"]
    chroma_token = st.secrets["CHROMA_CLOUD_TOKEN"]

    # Connect to Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)

    # Connect to Chroma Cloud
    client = chromadb.HttpClient(
        host=chroma_host,
        headers={"Authorization": f"Bearer {chroma_token}"}
    )
    
    # Connect the Vector Store
    vector_db = Chroma(
        client=client,
        collection_name="upsc_collection", # Make sure this matches your Chroma Cloud name
        embedding_function=embeddings
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# 3. FRONTEND
st.title("🎓 UPSCgtp")
st.caption("Connected to Cloud Intelligence")

with st.spinner("Connecting to Cloud..."):
    upsc_engine = init_upsc_cloud()

query = st.chat_input("Ask about UPSC topics...")

if query:
    st.chat_message("user").write(query)
    with st.spinner("Retrieving from Cloud Database..."):
        response = upsc_engine.invoke({"query": query})
        st.chat_message("assistant").write(response["result"])
