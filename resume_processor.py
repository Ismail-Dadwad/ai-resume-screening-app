import os
from dotenv import load_dotenv

# Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_chroma import Chroma

# Self-query & schema tools
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator

# ========== Gemini: Chat + Embeddings ========
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ------------------ Keys --------
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")

# ------------- Models -------------
# Embeddings: Gemini Embedding 001
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Chat LLM: override with GEMINI_MODEL in .env (each model has its own free-tier daily quota)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, max_retries=2)

# ----------- Functions ------------
# Load resumes in different formats
def load_resume(file_path):
    path = file_path.lower()
    if path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# Analyze the whole resume against the job description in a single Gemini call
def analyze_resume(docs, job_description):
    resume_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Compare this resume with the job description. Give:
1. Suitability Score (out of 100)
2. Skills Matched
3. Experience Relevance
4. Education Evaluation
5. Strengths
6. Weaknesses
7. Final Recommendation

Job Description:
{job_description}

Resume:
{resume_text}

"""
    result = llm.invoke(prompt)  # Gemini call
    return str(result.text)

# Split the resume into chunks and store them in ChromaDB, tagged with the resume's file name
def store_to_vectorstore(docs, resume_name, persist_directory="chroma_store"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"source": resume_name} for _ in texts]
    # Stable ids so re-uploading the same resume overwrites instead of duplicating
    ids = [f"{resume_name}_{i}" for i in range(len(texts))]

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return vectorstore

# Use SelfQueryRetriever to interpret and fetch relevant chunks
def run_self_query(query, persist_directory="chroma_store"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="File name of the resume the chunk comes from",
            type="string"
        )
    ]

    document_content_description = "A chunk of text from a candidate's resume."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        # Passed explicitly: auto-detection breaks with current langchain_community
        structured_query_translator=ChromaTranslator(),
        search_type="mmr"
    )
    return retriever.invoke(query)

# -------------------- End of resume_processor.py ----------------
