import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Faiza AI", page_icon="🤖", layout="centered")

st.title("🤖 Faiza AI")
st.caption("Faizan's Personal AI Assistant")

# ─── Prompt ────────────────────────────────────────────────────
custom_prompt_template = """You are Faiza, the personal AI assistant of Faizan.

Use ONLY the retrieved context from the vector database.

If the answer is present in the retrieved context, answer accurately.

If the answer is not present in the retrieved context, reply exactly:
"I don't have that information in my knowledge base."

Use external knowledge only for greetings like hi/hello.
Never guess. Never make assumptions. Never invent information.

If asked who made/created/developed/owns you:
Answer: "Faizan created and manages me."

If asked who/what you are:
Answer: "I am Faiza, Faizan's personal AI assistant."

Keep responses concise, professional, and based only on retrieved knowledge.

Context: {context}
Question: {input}
"""


# ─── Load Chain (cached) ────────────────────────────────────────
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "ageeegit/vector_store/db_faiss", embedding_model, allow_dangerous_deserialization=True
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=GROQ_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, document_chain)


qa_chain = load_chain()

# ─── Chat History ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat Input ─────────────────────────────────────────────────
user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"input": user_query})
            answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
