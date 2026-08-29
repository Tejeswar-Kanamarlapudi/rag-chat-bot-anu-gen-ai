import os

# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please create a .env file and add your API key."
    )


# ============================================================
# 2. LOAD DOCUMENTS
# ============================================================

print("\nLoading documents...")

loader = DirectoryLoader(
    "./documents",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")


# ============================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 5. CREATE CHROMA VECTOR DATABASE
# ============================================================

print("Creating vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database ready.")


# ============================================================
# 6. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 7. CREATE GROQ LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# ============================================================
# 8. CREATE RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful document-based assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say:

"I could not find this information in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 9. RAG CHATBOT LOOP
# ============================================================

print("\n" + "=" * 60)
print("           DOCUMENT-BASED RAG CHATBOT")
print("=" * 60)

print("\nType 'exit' or 'quit' to stop the chatbot.")

while True:

    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("\nChatbot stopped. Goodbye!")
        break

    if not question:
        continue

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieved_docs = retriever.invoke(question)

    print("\n" + "-" * 60)
    print("RETRIEVED CONTEXT")
    print("-" * 60)

    for i, doc in enumerate(retrieved_docs, start=1):

        print(f"\n[Chunk {i}]")
        print(doc.page_content)

    print("\n" + "-" * 60)

    # --------------------------------------------------------
    # COMBINE RETRIEVED CHUNKS
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # --------------------------------------------------------
    # SEND CONTEXT + QUESTION TO LLM
    # --------------------------------------------------------

    response = llm.invoke(messages)

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print("\nAnswer:")
    print(response.content[0]["text"])