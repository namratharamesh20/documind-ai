import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# --- Libraries ---
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq

# ----------------------------------------------------------
# CONFIGURATION & CSS (Dark Mode & "NotebookLM" Feel)
# ----------------------------------------------------------
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# Custom CSS for Dark Theme and Cards
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Cards */
    .stCard {
        background-color: #161B22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #E6EDF3;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2EA043;
    }

    /* Chat Messages */
    .user-msg {
        background-color: #1F6FEB;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .bot-msg {
        background-color: #21262D;
        color: #E6EDF3;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# ----------------------------------------------------------
# CLASSES & FUNCTIONS
# ----------------------------------------------------------

class FreeEmbeddings(Embeddings):
    """Local Embeddings using Sentence Transformers (Free)"""
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env")
        return None
    return Groq(api_key=api_key)

def llm_query(prompt, model="llama-3.3-70b-versatile", temperature=0.5):
    """Generic function to call Groq for Study Guide / Podcast"""
    client = get_groq_client()
    if not client: return "Error"
    
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------

def process_pdf(uploaded_file):
    """Reads PDF and prepares Vector DB + Full Text"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        # Load
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages]) # For summary/podcast features

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)

        # Vector Store
        embeddings = FreeEmbeddings()
        # Using a collection name helps separate different runs if needed
        vectordb = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="documind_collection" 
        )
        return vectordb, full_text
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

def create_rag_chain(vectordb):
    """Creates the Q&A Chain for Chat"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    template = """You are an intelligent research assistant. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise but informative.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | (lambda x: llm_query(x.to_messages()[0].content, temperature=0.1))
        | StrOutputParser()
    )
    return chain, retriever

# ----------------------------------------------------------
# SIDEBAR UI
# ----------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("DocuMind AI")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Source PDF", type=["pdf"])
    
    if uploaded_file and "vectordb" not in st.session_state:
        if st.button("🚀 Analyze Document"):
            with st.spinner("Ingesting Knowledge..."):
                vectordb, full_text = process_pdf(uploaded_file)
                chain, retriever = create_rag_chain(vectordb)
                
                # Store in session
                st.session_state.vectordb = vectordb
                st.session_state.rag_chain = chain
                st.session_state.retriever = retriever
                st.session_state.full_text = full_text # Save text for "Notebook" features
                
                st.success("Analysis Complete!")
                st.rerun()

    if "vectordb" in st.session_state:
        st.markdown("### 🛠 Options")
        if st.button("🧹 Clear Memory"):
            st.session_state.clear()
            st.rerun()

# ----------------------------------------------------------
# MAIN UI - TABS
# ----------------------------------------------------------
if "vectordb" in st.session_state:
    
    # Tabs for different text modes
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Study Guide", "🎙️ Podcast Script"])

    # --- TAB 1: CHAT ---
    with tab1:
        st.markdown("<div class='stCard'><h3>Chat with your Document</h3></div>", unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for message in st.session_state.messages:
            role_class = "user-msg" if message["role"] == "user" else "bot-msg"
            st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

        # Chat Input
        if prompt := st.chat_input("Ask something specific..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

            # Generate response
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f"<div class='bot-msg'>{response}</div>", unsafe_allow_html=True)

    # --- TAB 2: DEEP SUMMARY (Study Guide) ---
    with tab2:
        st.markdown("<div class='stCard'><h3>📄 Deep Summary & Study Guide</h3></div>", unsafe_allow_html=True)
        
        if st.button("✨ Generate Study Guide"):
            with st.spinner("Reading full document and summarizing..."):
                # We limit text to ~30k chars to stay safe within model limits while being fast
                safe_text = st.session_state.full_text[:30000] 
                
                summary_prompt = f"""
                Act as an expert professor. Analyze the following text and provide a comprehensive study guide.
                
                Structure the response as follows:
                1. 🎯 **Executive Summary** (1 paragraph)
                2. 🔑 **Key Concepts** (Bulleted list of core ideas)
                3. 📖 **Important Definitions** (Terminology explained)
                4. ❓ **Potential Quiz Questions** (3 questions based on the text)

                Text Content: {safe_text}
                """
                summary = llm_query(summary_prompt, model="llama-3.3-70b-versatile")
                st.markdown(summary)

    # --- TAB 3: PODCAST SCRIPT (Text Only) ---
    with tab3:
        st.markdown("<div class='stCard'><h3>📜 Generated Podcast Dialogue</h3></div>", unsafe_allow_html=True)
        st.caption("A generated script between two AI hosts discussing your PDF.")

        if st.button("🎙️ Write Podcast Script"):
            with st.spinner("Drafting script..."):
                safe_text = st.session_state.full_text[:30000]
                
                podcast_prompt = f"""
                Based on the text provided, write a podcast script between two hosts:
                1. **Alex** (The enthusiastic host who introduces topics).
                2. **Sam** (The analytical expert who explains details).

                They are discussing the main ideas of this document for a general audience.
                Make it conversational, engaging, and easy to understand. Use emojis where appropriate.
                
                Text Content: {safe_text}
                """
                script = llm_query(podcast_prompt, model="llama-3.3-70b-versatile")
                st.session_state.podcast_script = script
                st.rerun()

        if "podcast_script" in st.session_state:
            st.markdown("#### Script:")
            st.markdown(st.session_state.podcast_script)

else:
    # Landing Page
    st.markdown("""
    <div style="text-align: center; padding-top: 50px;">
        <h1 style="font-size: 3em;">🧠 DocuMind AI</h1>
        <p style="font-size: 1.2em; color: #8b949e;">Your Personal AI Research Assistant</p>
        <div style="background-color: #161B22; padding: 20px; border-radius: 10px; border: 1px solid #30363D; display: inline-block; text-align: left; margin-top: 20px;">
            <p><strong>✨ Features:</strong></p>
            <ul>
                <li>Chat with Documents (RAG)</li>
                <li>Instant Study Guides</li>
                <li>Podcast-style Script Generation</li>
            </ul>
        </div>
        <p style="margin-top: 20px;">⬅️ <em>Upload a PDF in the sidebar to get started</em></p>
    </div>
    """, unsafe_allow_html=True)