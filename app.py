import streamlit as st
import requests
import os
import json
import torch
import tempfile
import numpy as np
import faiss

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Voice
import speech_recognition as sr
from gtts import gTTS

# Translation
from deep_translator import GoogleTranslator

# PDF Export
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_LEFT

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="NVIDIA Multi PDF AI",
    page_icon="📄",
    layout="wide"
)

# ---------------- LOAD ENV ---------------- #
load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ---------------- LANGUAGE SUPPORT ---------------- #
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Chinese": "zh-CN"
}

# ---------------- SESSION STATE ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# ---------------- EMBEDDING MODEL ---------------- #
@st.cache_resource
def load_embedding_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    return model


embed_model = load_embedding_model()

# ---------------- SIDEBAR ---------------- #
with st.sidebar:

    st.title("⚙️ Settings")

    selected_language = st.selectbox(
        "🌍 Select Language",
        list(LANGUAGES.keys())
    )

    TARGET_LANG = LANGUAGES[selected_language]

    st.divider()

    st.title("📂 Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True
    )

# ---------------- TRANSLATION ---------------- #
def translate_to_english(text):

    if TARGET_LANG == "en":
        return text

    try:

        translated = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(text)

        return translated

    except:
        return text

# ---------------- PDF READING ---------------- #
def extract_pdf_text(pdf_docs):

    pages = []

    for pdf in pdf_docs:

        reader = PdfReader(pdf)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                pages.append({
                    "text": text,
                    "page": page_num + 1,
                    "source": pdf.name
                })

    return pages

# ---------------- CHUNKING ---------------- #
def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    )

    chunks = []

    for page in pages:

        split_texts = splitter.split_text(
            page["text"]
        )

        for chunk in split_texts:

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"]
            })

    return chunks

# ---------------- VECTOR STORE ---------------- #
@st.cache_resource
def build_vector_store(texts):

    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        32
    )

    index.add(embeddings)

    return index

# ---------------- SEARCH ---------------- #
def search_similar_chunks(
    question,
    index,
    chunks,
    k=10
):

    question_embedding = embed_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):

            retrieved_chunks.append(
                chunks[idx]
            )

    return retrieved_chunks

# ---------------- NVIDIA AI ---------------- #
def ask_ai(question, context):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""
You are an advanced multilingual AI assistant.

IMPORTANT:
- Respond ONLY in this language: {TARGET_LANG}
- Never answer in English unless language is English.
- Use natural fluent language.
- Answer ONLY from provided PDF context.
- Give highly detailed structured answers.
- Use headings, bullet points, examples,
  and step-by-step explanations.
"""

    payload = {
        "model": "meta/llama-3.1-70b-instruct",

        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],

        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 2500,
        "stream": True
    }

    try:

        response = requests.post(
            URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=180
        )

        full_answer = ""

        placeholder = st.empty()

        for line in response.iter_lines():

            if line:

                decoded_line = line.decode("utf-8")

                if decoded_line.startswith("data: "):

                    data = decoded_line[6:]

                    if data == "[DONE]":
                        break

                    try:

                        json_data = json.loads(data)

                        delta = (
                            json_data["choices"][0]
                            .get("delta", {})
                        )

                        if "content" in delta:

                            token = delta["content"]

                            full_answer += token

                            placeholder.markdown(
                                full_answer + "▌"
                            )

                    except:
                        pass

        placeholder.markdown(full_answer)

        return full_answer

    except Exception as e:

        return f"❌ Error: {str(e)}"

# ---------------- VOICE INPUT ---------------- #
def get_voice_input():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Listening...")

        recognizer.adjust_for_ambient_noise(
            source
        )

        audio = recognizer.listen(
            source,
            phrase_time_limit=10
        )

    try:

        text = recognizer.recognize_google(
            audio,
            language=TARGET_LANG
        )

        return text

    except Exception as e:

        st.error(f"Voice Error: {e}")

        return ""

# ---------------- TEXT TO SPEECH ---------------- #
def text_to_speech(text):

    tts = gTTS(
        text=text,
        lang=TARGET_LANG
    )

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(temp_audio.name)

    return temp_audio.name

# ---------------- PDF EXPORT ---------------- #
def generate_pdf(answer):

    pdf_path = "answer.pdf"

    # Register Unicode Font
    font_path = "NotoSans-Regular.ttf"

    pdfmetrics.registerFont(
        TTFont("NotoSans", font_path)
    )

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    unicode_style = ParagraphStyle(
        "Unicode",
        parent=styles["BodyText"],
        fontName="NotoSans",
        fontSize=11,
        leading=18,
        alignment=TA_LEFT
    )

    story = []

    for line in answer.split("\n"):

        if line.strip():

            story.append(
                Paragraph(
                    line,
                    unicode_style
                )
            )

            story.append(
                Spacer(1, 8)
            )

    doc.build(story)

    return pdf_path

# ---------------- PROCESS PDFS ---------------- #
with st.sidebar:

    if st.button("⚡ Process PDFs"):

        if pdf_docs:

            with st.spinner(
                "Processing PDFs..."
            ):

                pages = extract_pdf_text(
                    pdf_docs
                )

                chunks = create_chunks(
                    pages
                )

                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                vector_store = build_vector_store(
                    texts
                )

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.chunks = (
                    chunks
                )

            st.success(
                "✅ PDFs Processed Successfully!"
            )

        else:

            st.warning(
                "Upload PDFs first"
            )

# ---------------- MAIN UI ---------------- #
st.title(
    "📄 NVIDIA AI Multi PDF Chatbot"
)

st.write(
    "Ask questions from PDFs in any language"
)

# ---------------- VOICE BUTTON ---------------- #
if st.button("🎤 Voice Question"):

    voice_question = get_voice_input()

    if voice_question:

        st.session_state.voice_question = (
            voice_question
        )

# ---------------- CHAT INPUT ---------------- #
question = st.chat_input(
    "Ask anything from PDFs..."
)

# Voice Question
if "voice_question" in st.session_state:

    question = (
        st.session_state.voice_question
    )

    del st.session_state.voice_question

# ---------------- PROCESS QUESTION ---------------- #
if question:

    if st.session_state.vector_store is None:

        st.warning(
            "⚠️ Upload and process PDFs first"
        )

    else:

        st.session_state.chat_history.append(
            ("user", question)
        )

        with st.spinner(
            "Searching PDFs..."
        ):

            english_question = (
                translate_to_english(
                    question
                )
            )

            retrieved_chunks = (
                search_similar_chunks(
                    english_question,
                    st.session_state.vector_store,
                    st.session_state.chunks
                )
            )

            context = "\n\n".join(
                [
                    chunk["text"][:2500]
                    for chunk in retrieved_chunks
                ]
            )

            sources = []

            for chunk in retrieved_chunks:

                source_text = (
                    f"{chunk['source']} "
                    f"- Page {chunk['page']}"
                )

                if source_text not in sources:

                    sources.append(
                        source_text
                    )

        with st.chat_message("assistant"):

            answer = ask_ai(
                english_question,
                context
            )

            # Backup translation
            if TARGET_LANG != "en":

                try:

                    answer = GoogleTranslator(
                        source='auto',
                        target=TARGET_LANG
                    ).translate(answer)

                except:
                    pass

            source_text = "\n\n📚 Sources:\n"

            for src in sources:

                source_text += f"- {src}\n"

            final_answer = (
                answer + source_text
            )

            st.write(final_answer)

            # Save Audio
            audio_path = text_to_speech(answer)

            st.session_state.audio_file = audio_path

        st.session_state.chat_history.append(
            (
                "assistant",
                final_answer
            )
        )

# ---------------- DISPLAY CHAT HISTORY ---------------- #
for role, message in st.session_state.chat_history:

    with st.chat_message(role):

        st.write(message)

# ---------------- AUDIO PLAYER ---------------- #
if st.session_state.audio_file:

    st.audio(
        st.session_state.audio_file
    )

# ---------------- DOWNLOADS ---------------- #
if st.session_state.chat_history:

    last_message = (
        st.session_state.chat_history[-1][1]
    )

    st.download_button(
        "📥 Download TXT",
        last_message,
        file_name="answer.txt"
    )

    pdf_file = generate_pdf(
        last_message
    )

    with open(pdf_file, "rb") as f:

        st.download_button(
            "📥 Download PDF",
            f,
            file_name="answer.pdf"
        )
