from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,  # ← Sirf yahi badla
    )
    documents = loader.load()
    return documents


documents = load_pdf_files(data=DATA_PATH)


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks


chunks = create_chunks(extracted_data=documents)


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()

DB_PATH = "vector_store/db_faiss"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_PATH)
