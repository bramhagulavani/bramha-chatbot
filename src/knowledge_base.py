import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Step 1 — Load your personal knowledge file ───────────────────────────────
print("Loading personal knowledge base...")
loader = TextLoader("data/personal_knowledge.txt", encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)!")

# ── Step 2 — Split into small chunks ────────────────────────────────────────
# Simple explanation: We cut the big text file into small pieces
# so the AI can find exactly the right piece to answer each question
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk = 500 characters
    chunk_overlap=50      # 50 character overlap between chunks
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks!")

# ── Step 3 — Create embeddings and store in FAISS ───────────────────────────
# Simple explanation: embeddings convert text into numbers
# so the AI can search by meaning, not just keywords
print("Creating embeddings — this takes 1-2 minutes first time...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)
print("Knowledge base ready! ✅")

# ── Step 4 — Save vector store locally ──────────────────────────────────────
os.makedirs("models/embeddings", exist_ok=True)
vector_store.save_local("models/embeddings")
print("Knowledge base saved to models/embeddings! ✅")

# ── Step 5 — Search function ─────────────────────────────────────────────────
def search_knowledge(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

# ── Step 6 — RAG chat function ───────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Bramha's personal AI assistant.
You have access to Bramha's personal knowledge base.
Always use the provided context to answer questions about Bramha.
Be friendly, encouraging and use emojis occasionally.
Never say you are ChatGPT or any other AI.
"""

chat_history = []

def rag_chat(user_message):
    # Search knowledge base for relevant context
    context = search_knowledge(user_message)

    # Build message with context injected
    message_with_context = f"""
Context from Bramha's knowledge base:
{context}

User question: {user_message}
"""

    chat_history.append({
        "role": "user",
        "content": message_with_context
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + chat_history,
        max_tokens=500
    )

    ai_reply = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply

# ── Test RAG ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   RAG TEST — ANSWERING FROM KNOWLEDGE BASE")
print("=" * 50)

test_questions = [
    "What are Bramha's skills?",
    "Tell me about the Plant Disease Detector project",
    "What are Bramha's goals?",
    "What subjects does Bramha study?"
]

for question in test_questions:
    print(f"\n🧑 Bramha: {question}")
    reply = rag_chat(question)
    print(f"🤖 AI: {reply}")
    print("-" * 50)