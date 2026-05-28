import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Personality System Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Bramha's personal AI assistant — a smart, friendly and helpful chatbot
built specifically for Bramha Vinayak Gulavani.

## About Bramha:
- Full name: Bramha Vinayak Gulavani
- College: VIT Pune (Vishwakarma Institute of Technology)
- Year: Second Year, Second Semester
- Branch: AI & ML Engineering
- City: Pune, Maharashtra, India

## Bramha's AI Projects:
1. Plant Disease Detector — MobileNetV2, 94.16% accuracy
2. Pune Rent Predictor — XGBoost, 74.79% R²
3. Bramha Chatbot — LLaMA 3.3 via Groq (this project!)

## Your Personality:
- Always address the user as Bramha
- Be friendly, encouraging and motivating
- Keep responses clear and simple
- Use emojis occasionally to keep tone fun
- Never say you are ChatGPT or any other AI
- You are Bramha's personal AI — act like it!
"""

# ── Load Knowledge Base ──────────────────────────────────────────────────────
print("Loading knowledge base...")

# Check if embeddings already saved — if yes, load directly
if os.path.exists("models/embeddings/index.faiss"):
    print("Loading existing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "models/embeddings",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Knowledge base loaded! ✅")
else:
    print("Creating embeddings for first time...")
    loader = TextLoader("data/personal_knowledge.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs("models/embeddings", exist_ok=True)
    vector_store.save_local("models/embeddings")
    print("Knowledge base created and saved! ✅")

# ── Search Knowledge Base ────────────────────────────────────────────────────
def search_knowledge(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

# ── Main Chat Function ───────────────────────────────────────────────────────
def chat(message, history):
    # Search knowledge base for relevant context
    context = search_knowledge(message)

    # Build message with context
    message_with_context = f"""
Context from Bramha's personal knowledge base:
{context}

Bramha's message: {message}
"""

    # Convert Gradio history to API format
    api_history = []
    for human, assistant in history:
        api_history.append({"role": "user", "content": human})
        api_history.append({"role": "assistant", "content": assistant})

    # Add current message
    api_history.append({
        "role": "user",
        "content": message_with_context
    })

    # Get response from LLaMA 3.3
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + api_history,
        max_tokens=600
    )

    return response.choices[0].message.content

# ── Custom CSS — WhatsApp Dark Theme ────────────────────────────────────────
custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
        min-height: 100vh;
    }
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-align: center;
    }
    .block {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
    }
    button.primary {
        background: linear-gradient(90deg, #56ab2f, #a8e063) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #0f2027 !important;
        font-weight: 700 !important;
    }
    footer { display: none !important; }
"""

# ── Banner HTML ──────────────────────────────────────────────────────────────
banner = """
<div style="text-align:center; padding:10px 0 18px;">
    <p style="color:#a8d5a2; font-size:1rem; margin:0;">
        🧠 Powered by <b>LLaMA 3.3</b> via Groq &nbsp;|&nbsp;
        📚 Personal Knowledge Base &nbsp;|&nbsp;
        🧠 Conversation Memory &nbsp;|&nbsp;
        ⚡ Under 1 second response
    </p>
    <p style="color:#7fb3a0; font-size:0.85rem; margin-top:6px;">
        Your personal AI — knows who you are, remembers what you said!
    </p>
</div>
"""

footer_html = """
<div style="text-align:center; margin-top:16px; color:#7fb3a0; font-size:0.82rem;">
    Built by <b style="color:#a8d5a2;">Bramha Vinayak Gulavani</b> &nbsp;·&nbsp;
    AI & ML Student, VIT Pune &nbsp;·&nbsp;
    <a href="https://github.com/bramhagulavani/bramha-chatbot"
       style="color:#56ab2f;" target="_blank">GitHub →</a>
</div>
"""

# ── Build Gradio Chat App ────────────────────────────────────────────────────
with gr.Blocks(title="🤖 Bramha's Personal AI") as app:

    gr.HTML("<h1>🤖 Bramha's Personal AI</h1>")
    gr.HTML(banner)

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
    height=450,
    avatar_images=(
                "https://api.dicebear.com/7.x/initials/svg?seed=BG&backgroundColor=378ADD",
                "https://api.dicebear.com/7.x/bottts/svg?seed=bramha"
            )
        ),
        textbox=gr.Textbox(
            placeholder="Ask me anything, Bramha...",
            container=False,
            scale=7
        ),
        examples=[
            "Who are you?",
            "What projects have I built?",
            "What are my skills?",
            "Give me motivation for today!",
            "What are my goals?"
        ],
        cache_examples=False,
    )

    gr.HTML(footer_html)

# ── Launch ───────────────────────────────────────────────────────────────────
print("Starting Bramha's Personal AI...")
app.launch(css=custom_css)