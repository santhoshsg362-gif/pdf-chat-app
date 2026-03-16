import streamlit as st
import os
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from gtts import gTTS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- LOAD API ----------------

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NVIDIA AI PDF Assistant",
    layout="wide"
)


# ---------------- NVIDIA STYLE UI ----------------

st.markdown("""
<style>

.stApp {
background-color:#0e0e0e;
color:white;
}

h1 {
color:#76B900;
text-align:center;
}

section[data-testid="stSidebar"] {
background-color:#111111;
}

.stButton>button {
background-color:#76B900;
color:black;
border-radius:8px;
border:none;
font-weight:bold;
}

.stButton>button:hover {
background-color:#5fa300;
}

[data-testid="stChatMessage"] {
background-color:#1a1a1a;
border-radius:10px;
padding:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.image(
"https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg",
width=180
)

st.markdown(
"""
<h1>⚡ NVIDIA AI PDF Assistant</h1>
<p style='text-align:center;color:gray'>
Chat with your PDFs using AI
</p>
""",
unsafe_allow_html=True
)


# ---------------- PDF TEXT EXTRACTION ----------------

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            reader = PdfReader(pdf)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text

        except:

            st.error("Invalid or corrupted PDF file.")
            return ""

    return text


# ---------------- VECTOR STORE ----------------

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store


# ---------------- NVIDIA AI CALL ----------------

def ask_ai(context, question):

    prompt = f"""
You are an AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen/qwen3.5-397b-a17b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]

    except:
        return "AI response error."


# ---------------- TEXT TO SPEECH ----------------

def text_to_speech(text):

    tts = gTTS(text=text)

    file = "response.mp3"

    tts.save(file)

    return file


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Process PDFs"):

        if pdf_docs:

            text = get_pdf_text(pdf_docs)

            if text:

                vector_store = create_vector_store(text)

                st.session_state.vector_store = vector_store

                st.success("PDF processed successfully!")

        else:

            st.warning("Please upload a PDF first.")


# ---------------- CHAT MEMORY ----------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display chat history

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

question = st.chat_input("Ask something about the PDF")


# ---------------- AI RESPONSE ----------------

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):

        st.markdown(question)

    if "vector_store" not in st.session_state:

        answer = "Please upload and process a PDF first."

    else:

        docs = st.session_state.vector_store.similarity_search(question)

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        answer = ask_ai(context, question)

    with st.chat_message("assistant"):

        st.markdown(answer)

        audio = text_to_speech(answer)

        st.audio(audio)

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
