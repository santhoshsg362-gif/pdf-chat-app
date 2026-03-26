import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import speech_recognition as sr
from gtts import gTTS
import tempfile

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="PDF Chat AI", layout="wide")

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- PDF PROCESSING ---------------- #
def get_pdf_text_with_page(pdf_docs):
    texts = []

    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            texts.append({
                "text": page_text,
                "page": i + 1
            })

    return texts


def get_text_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = []

    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for chunk in split_texts:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


# ---------------- FAISS ---------------- #
def create_vector_store(chunks):
    texts = [c["text"] for c in chunks]

    embeddings = embed_model.encode(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, chunks


def search_chunks(question, index, chunks, k=20):
    q_embedding = embed_model.encode([question])
    distances, indices = index.search(np.array(q_embedding), k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results


# ---------------- NVIDIA AI ---------------- #
def ask_ai(context, question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta/llama3-70b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. "
                    "Give very detailed, structured answers. "
                    "Explain step-by-step. "
                    "Use headings, bullet points, and examples. "
                    "Do not give short answers."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        "temperature": 0.6,
        "max_tokens": 3000
    }

    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error: {response.text}"

    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return response.text


# ---------------- VOICE ---------------- #
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)

    try:
        return r.recognize_google(audio)
    except:
        st.error("Voice error")
        return ""


def speak(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


# ---------------- PDF DOWNLOAD ---------------- #
def create_pdf(text):
    file_path = "answer.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    for line in text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))

    doc.build(content)
    return file_path


# ---------------- SESSION STATE ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "audio_played" not in st.session_state:
    st.session_state.audio_played = False


# ---------------- UI ---------------- #
st.title("📄 Chat with PDFs NVIDIA AI")

# Sidebar
with st.sidebar:
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing..."):
                pages = get_pdf_text_with_page(pdf_docs)
                chunks = get_text_chunks(pages)

                index, chunks = create_vector_store(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks

            st.success("✅ File is Ready!")
        else:
            st.warning("Upload PDFs")


# 🎤 Voice
if st.button("🎤 Speak"):
    voice_q = get_voice_input()
    if voice_q:
        st.session_state.voice_question = voice_q

# 💬 Chat input
question = st.chat_input("Ask from your PDFs")

# Merge voice + text
if "voice_question" in st.session_state and not question:
    question = st.session_state.voice_question
    del st.session_state.voice_question


# ---------------- MAIN LOGIC ---------------- #
if question:
    if "index" not in st.session_state:
        st.warning("⚠️ Upload PDFs first")
    else:
        # Save user message
        st.session_state.chat_history.append(("user", question))

        relevant_chunks = search_chunks(
            question,
            st.session_state.index,
            st.session_state.chunks,
            k=20
        )

        context = " ".join([c["text"] for c in relevant_chunks])

        # 📄 Collect page numbers
        pages_used = sorted(set([c["page"] for c in relevant_chunks]))

        with st.spinner("Thinking..."):
            answer = ask_ai(context, question)

        # Add page reference
        page_info = f"\n\n📄 Sources: Pages {', '.join(map(str, pages_used))}"
        final_answer = answer + page_info

        # Save answer
        st.session_state.chat_history.append(("assistant", final_answer))

        # Voice
        audio_file = speak(answer)
        st.session_state.audio_file = audio_file
        st.session_state.audio_played = False


# ---------------- DISPLAY CHAT ---------------- #
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)


# 🔊 Play audio once
if "audio_file" in st.session_state and not st.session_state.audio_played:
    st.audio(st.session_state.audio_file)
    st.session_state.audio_played = True


# 📥 Downloads
if st.session_state.chat_history:
    last_answer = st.session_state.chat_history[-1][1]

    st.download_button("📥 Download TXT", last_answer, "answer.txt")

    pdf = create_pdf(last_answer)
    with open(pdf, "rb") as f:
        st.download_button("📥 Download PDF", f, "answer.pdf")
