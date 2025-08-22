import os
import re
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from langdetect import detect
from streamlit_mic_recorder import speech_to_text
from bott_backend import get_bot_response  # Backend function

from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# =========================
# Helpers
# =========================
def _clean(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s\u0600-\u06FF\.\,\?\!\:\;\'\-\(\)]+", " ", text).strip()

def _detect_lang(text: str) -> str:
    try:
        lang = detect(text or "")
        return "ur" if lang == "ur" else "en"
    except Exception:
        return "en"

def tts_bytes(text: str) -> tuple[BytesIO, str]:
    lang = _detect_lang(text)
    mp3 = BytesIO()
    gTTS(text=_clean(text), lang=lang, slow=False).write_to_fp(mp3)
    mp3.seek(0)
    return mp3, lang

# =========================
# Load API Key
# =========================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY in .env file")
    st.stop()

# =========================
# Page Config + CSS
# =========================
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")
st.markdown("""
<style>
.main .block-container { padding-top: 1.2rem; padding-bottom: 7rem; }
.chat-wrap { max-width: 900px; margin: 0 auto; }
.user-bubble { background: linear-gradient(120deg, #a8e6cf, #dcedc1); color: #000; padding: 8px 14px; border-radius: 18px 18px 0 18px; margin:4px 0; max-width:70%; float:right; clear:both; }
.bot-bubble { background: linear-gradient(120deg, #10b981, #3b82f6); color: #fff; padding:8px 14px; border-radius:18px 18px 18px 0; margin:4px 0; max-width:70%; float:left; clear:both; }
.dock-bottom { position: fixed; left:0; right:0; bottom:0; background: linear-gradient(180deg, rgba(17,24,39,0), rgba(17,24,39,0.75) 30%, rgba(17,24,39,0.95)); padding:12px 16px; backdrop-filter: blur(6px); border-top:1px solid rgba(255,255,255,0.08); }
.dock-inner { max-width:900px; margin:0 auto; display:flex; align-items:center; }
.icon-btn button { font-size:1.6rem !important; padding:0.4rem 0.7rem !important; border-radius:50% !important; cursor:pointer; transition: transform 0.2s, background-color 0.2s; background-color: rgba(255,255,255,0.05) !important; border:none; }
.icon-btn button:hover { transform: scale(1.3); background-color: rgba(255,255,255,0.15) !important; }
.stChatMessage { padding:0.2rem 0; }
.stChatMessage .stMarkdown { font-size:1rem; line-height:1.55; }
footer { visibility:hidden; height:0; }
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
tts_enable = st.sidebar.toggle("🔊 Speak bot replies", value=True)
auto_send_voice = st.sidebar.toggle("🎤 Auto-send mic input", value=True)

if st.sidebar.button("🧹 Clear Chat"):
    for k in ["messages","last_bot","prefill_text","last_input_type"]:
        if k in st.session_state: st.session_state.pop(k)

# =========================
# LLM Setup
# =========================
llm = ChatGroq(temperature=temperature, groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful chatbot. Keep replies short unless user asks for detail."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# =========================
# Session State
# =========================
if "messages" not in st.session_state: st.session_state.messages=[]
if "last_bot" not in st.session_state: st.session_state.last_bot=None
if "prefill_text" not in st.session_state: st.session_state.prefill_text=""
if "last_input_type" not in st.session_state: st.session_state.last_input_type="text"

def run_chat(user_text: str, input_type: str):
    if not user_text.strip(): return
    st.session_state.messages.append({"role":"user","content":user_text})
    bot_text = get_bot_response(user_text)  # Backend integrated here
    st.session_state.messages.append({"role":"assistant","content":bot_text})
    st.session_state.last_bot = bot_text
    st.session_state.prefill_text=""
    st.session_state.last_input_type=input_type

# =========================
# Header + Messages
# =========================
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
st.title("🤖 Conversational Chatbot")
for msg in st.session_state.messages:
    cls = "user-bubble" if msg["role"]=="user" else "bot-bubble"
    st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Docked Chat Bar
# =========================
with st.container():
    st.markdown('<div class="dock-bottom"><div class="dock-inner">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([8,1,1])
    
    # Text input with on_change auto-send
    def send_text():
        if st.session_state.prefill_text.strip():
            run_chat(st.session_state.prefill_text, "text")
            st.session_state.prefill_text = ""

    with col1:
        user_input = st.text_input(
            "Type your message...", 
            value=st.session_state.prefill_text,
            label_visibility="collapsed",
            key="prefill_text",
            on_change=send_text
        )
    
    # Voice input button
    with col2:
        if st.button("🎤", key="voice_btn", help="Record voice"):
            voice_text = speech_to_text(language="en-US", use_container_width=True, just_once=True)
            if voice_text:
                if auto_send_voice:
                    run_chat(voice_text, "voice")
                else:
                    st.session_state.prefill_text = voice_text
                    st.session_state.last_input_type = "voice"
    
    # Send text button fallback
    with col3:
        if st.button("📤") and st.session_state.prefill_text.strip():
            run_chat(st.session_state.prefill_text, "text")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================
# TTS Playback only for voice input
# =========================
if tts_enable and st.session_state.last_bot and st.session_state.last_input_type=="voice":
    try:
        audio_mp3, lang_code = tts_bytes(st.session_state.last_bot)
        st.audio(audio_mp3, format="audio/mp3", start_time=0)
        st.caption(f"🗣 Bot reply spoken in {lang_code.upper()}")
    except Exception as e:
        st.info(f"Note: Could not play audio ({e}).")
