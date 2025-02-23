import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# App Title
st.title('📄 RAG-DOC-Chat')

# File Upload
uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# API Key Input
groq_api_key = st.text_input("🔑 Enter your Groq API key:", type="password")

# Check for valid API key
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key to proceed.")
    st.stop()

# Process PDF (Only once)
if uploaded_file is not None and not st.session_state.pdf_processed:
    with open('uploaded_file.pdf', "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF and Split into Chunks
    loader = PyPDFLoader('uploaded_file.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Convert to Embeddings and Store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory="./chroma_db"
    )

    # Mark PDF as Processed
    st.session_state.pdf_processed = True

# Load LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# Define System Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use ten sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

# Create Chat Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create Document Processing Chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Set up RAG Retrieval Chain if not already initialized
if st.session_state.vectorstore and st.session_state.rag_chain is None:
    retriever = st.session_state.vectorstore.as_retriever()
    st.session_state.rag_chain = create_retrieval_chain(retriever, document_chain)

# Display Chat History
st.write("### 💬 Chat History")
for i, (input_text, output_text) in enumerate(st.session_state.chat_history):
    st.write(f"**🧑 You:** {input_text}")
    st.write(f"**🤖 Assistant:** {output_text}")
    st.write("---")

# User Input for Chat
user_input = st.text_input("📝 Ask a question about the PDF:", key="user_input")

# Process user input when button is clicked
if st.button("Send") and user_input:
    # Get response from RAG model
    results = st.session_state.rag_chain.invoke({"input": user_input})
    answer = results["answer"]

    # Append conversation to chat history
    st.session_state.chat_history.append((user_input, answer))

    # Clear the text input without modifying session state
    user_input = ''

# Display last assistant response
if st.session_state.chat_history:
    last_answer = st.session_state.chat_history[-1][1]
    st.write(f"**🤖 Assistant:** {last_answer}")
