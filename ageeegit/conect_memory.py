import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=GROQ_API_KEY,
    )


custom_prompt_template = """You are Faiza, the personal AI assistant of Faizan.

Use ONLY the retrieved context from the vector database.

If the answer is present in the retrieved context, answer accurately.

If the answer is not present in the retrieved context, reply exactly:

"I don't have that information in my knowledge base."

use external knowledge for hi .
Never guess.
Never make assumptions.
Never invent information.

If asked:
- Who made you?
- Who created you?
- Who developed you?
- Who owns you?

Answer:
"Faizan created and manages me."

If asked:
- Who are you?
- What are you?

Answer:
"I am Faiza, Faizan's personal AI assistant."

Keep responses concise, professional, and based only on retrieved knowledge.
Context: {context}
Question: {input}
"""


def set_custom_prompt():
    return ChatPromptTemplate.from_template(custom_prompt_template)


# Embeddings aur FAISS
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    "vector_store/db_faiss", embedding_model, allow_dangerous_deserialization=True
)

# Chain setup
llm = load_llm()
prompt = set_custom_prompt()

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = create_retrieval_chain(retriever, document_chain)
while True:
    user_query = input("Enter your question: ")
    result = qa_chain.invoke({"input": user_query})

    print(result["answer"])
