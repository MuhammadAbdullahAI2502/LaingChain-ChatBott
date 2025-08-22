# bott_backend.py
import os
from dotenv import load_dotenv
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
# Load API Key
# =========================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ Missing GROQ_API_KEY in .env file")

# =========================
# LLM Setup
# =========================
llm = ChatGroq(
    temperature=0.7,  # Default; can be overridden from frontend
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful chatbot. Keep replies short unless user asks for detail."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

# Conversation memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Conversation chain
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# =========================
# Main backend function
# =========================
def get_bot_response(user_text: str) -> str:
    """
    This function receives user input text and returns the bot's response
    using the Groq LLM via LangChain.
    """
    if not user_text.strip():
        return ""
    bot_reply = conversation.predict(input=user_text)
    return bot_reply
