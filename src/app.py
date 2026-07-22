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
    try:
        results = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"Knowledge search warning: {e}")
        return ""

# ── Main Chat Function ───────────────────────────────────────────────────────
def chat(message, history):
    # Search knowledge base for relevant context
    context = search_knowledge(message)

    # Build message with context injected
    message_with_context = f"""
Context from Bramha's personal knowledge base:
{context}

Bramha's message: {message}
"""

    # Convert Gradio history format to API format
    api_history = []
    for msg in history:
        if isinstance(msg, dict):
            # Gradio modern dict format
            api_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Legacy tuple format
            if msg[0]:
                api_history.append({"role": "user", "content": str(msg[0])})
            if msg[1]:
                api_history.append({"role": "assistant", "content": str(msg[1])})

    # Add current message with context
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

# ── Custom CSS — UI/UX Pro Max Glassmorphism ─────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Container & Master Theme */
.gradio-container {
    background: radial-gradient(circle at 10% 15%, rgba(16, 185, 129, 0.1) 0%, transparent 35%),
                radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.1) 0%, transparent 35%),
                linear-gradient(135deg, #070A11 0%, #0F172A 50%, #080D1A 100%) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #F3F4F6 !important;
    min-height: 100vh;
    padding: 16px 24px !important;
}

/* Glassmorphism Panels */
.glass-panel, .block, div[class*="gr-box"] {
    background: rgba(17, 24, 39, 0.72) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    box-shadow: 0 12px 32px -4px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
}

/* Header Banner */
.cyber-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.08)) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 24px !important;
    padding: 24px 30px !important;
    margin-bottom: 20px !important;
}

.cyber-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #FFFFFF 30%, #10B981 70%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 2s infinite;
}

@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.metric-pill {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 8px 12px;
    text-align: center;
}

.metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #38BDF8;
}

.metric-label {
    font-size: 0.7rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Developer Profile Sidebar */
.profile-card {
    padding: 18px;
    text-align: center;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 16px;
}

.profile-avatar {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 2px solid #10B981;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    margin: 0 auto 10px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    background: rgba(16, 185, 129, 0.1);
}

.profile-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F9FAFB;
    margin-bottom: 2px;
}

.profile-title {
    font-size: 0.82rem;
    color: #9CA3AF;
    margin-bottom: 10px;
}

.tech-tag {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}

.project-item {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.project-item:hover {
    border-color: rgba(16, 185, 129, 0.4);
    transform: translateX(3px);
}

.project-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #F3F4F6;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.project-badge {
    font-size: 0.68rem;
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    padding: 1px 6px;
    border-radius: 9999px;
    font-weight: 600;
}

.project-desc {
    font-size: 0.75rem;
    color: #9CA3AF;
    margin-top: 3px;
}

/* Chatbot Container & High-Contrast Message Styling */
.gradio-container .chatbot {
    background: rgba(15, 23, 42, 0.85) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

/* User Message Bubble - High Contrast */
.gradio-container .user,
.gradio-container [data-testid="user"],
.gradio-container .message.user,
.gradio-container .user-row .message {
    background: #1E293B !important;
    border: 1.5px solid #38BDF8 !important;
    color: #FFFFFF !important;
    border-radius: 16px 16px 4px 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
}

/* AI Bot Message Bubble - High Contrast */
.gradio-container .bot,
.gradio-container [data-testid="bot"],
.gradio-container .message.bot,
.gradio-container .bot-row .message {
    background: #0F172A !important;
    border: 1.5px solid #10B981 !important;
    border-left: 5px solid #10B981 !important;
    color: #FFFFFF !important;
    border-radius: 16px 16px 16px 4px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
}

/* Force ALL internal text elements in Chatbot to crisp, bright white */
.gradio-container .chatbot *,
.gradio-container .chatbot p,
.gradio-container .chatbot span,
.gradio-container .chatbot div,
.gradio-container .chatbot li,
.gradio-container .chatbot td,
.gradio-container .chatbot th,
.gradio-container .chatbot code,
.gradio-container .chatbot strong,
.gradio-container .chatbot em,
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .md,
.gradio-container .md * {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
}

/* Code Snippets & Blocks inside AI replies */
.gradio-container .chatbot pre,
.gradio-container .chatbot code {
    background: #070A11 !important;
    color: #34D399 !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    border-radius: 8px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* High Contrast Input Text Box & Placeholder */
.gradio-container textarea, 
.gradio-container input[type="text"],
.gradio-container input {
    background: #0F172A !important;
    border: 1.5px solid #38BDF8 !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
}

.gradio-container textarea::placeholder, 
.gradio-container input::placeholder {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

.gradio-container textarea:focus, 
.gradio-container input:focus {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3) !important;
    outline: none !important;
}

/* Example Prompt Chips Readability */
.gradio-container .examples,
.gradio-container button.example,
.gradio-container .example {
    background: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
}

.gradio-container button.example:hover,
.gradio-container .example:hover {
    background: rgba(16, 185, 129, 0.2) !important;
    color: #34D399 !important;
    border-color: #10B981 !important;
    transform: translateY(-1px) !important;
}

button.primary {
    background: linear-gradient(135deg, #10B981 0%, #06B6D4 100%) !important;
    color: #070A11 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
}

footer { display: none !important; }

.custom-footer {
    text-align: center;
    padding: 16px;
    margin-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #9CA3AF;
    font-size: 0.82rem;
}

.custom-footer a {
    color: #34D399;
    text-decoration: none;
    font-weight: 600;
}

.custom-footer a:hover {
    text-decoration: underline;
}
"""

# ── Header & Banner HTML Components ──────────────────────────────────────────
header_html = """
<div class="cyber-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="cyber-title">🤖 Bramha's Personal AI</h1>
            <p style="color: #9CA3AF; font-size: 0.92rem; margin: 0;">
                Smart Conversational Assistant powered by <b>LLaMA 3.3 (70B)</b> & FAISS RAG Memory
            </p>
        </div>
        <div class="status-badge">
            <span class="pulse-dot"></span>
            <span>SYSTEM OPERATIONAL</span>
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-pill">
            <div class="metric-value">LLaMA 3.3</div>
            <div class="metric-label">70B Model</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value">Groq Cloud</div>
            <div class="metric-label">&lt; 1s Latency</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value">FAISS RAG</div>
            <div class="metric-label">Vector Knowledge</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value">VIT Pune</div>
            <div class="metric-label">AI & ML Dept</div>
        </div>
    </div>
</div>
"""

sidebar_profile_html = """
<div class="profile-card">
    <div class="profile-avatar">👨‍💻</div>
    <div class="profile-name">Bramha Vinayak Gulavani</div>
    <div class="profile-title">AI & ML Engineering Student • VIT Pune</div>
    <div style="margin-bottom: 12px;">
        <span class="tech-tag">Python</span>
        <span class="tech-tag">TensorFlow</span>
        <span class="tech-tag">LangChain</span>
        <span class="tech-tag">Groq</span>
        <span class="tech-tag">FAISS</span>
    </div>
    <a href="https://github.com/bramhagulavani" target="_blank" 
       style="display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.06); color: #38BDF8; text-decoration: none; padding: 6px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.2);">
       🔗 GitHub Profile →
    </a>
</div>

<div style="margin-bottom: 16px;">
    <h4 style="color: #F3F4F6; font-size: 0.9rem; font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 6px;">
        🚀 Featured AI Projects
    </h4>
    
    <div class="project-item">
        <div class="project-title">
            <span>🌿 Plant Disease Detector</span>
            <span class="project-badge">94.16% Acc</span>
        </div>
        <div class="project-desc">MobileNetV2 Transfer Learning model trained on 54k leaf images.</div>
    </div>

    <div class="project-item">
        <div class="project-title">
            <span>🏠 Pune Rent Predictor</span>
            <span class="project-badge">74.79% R²</span>
        </div>
        <div class="project-desc">XGBoost ML Regressor for rental estimation across 343 Pune locations.</div>
    </div>

    <div class="project-item">
        <div class="project-title">
            <span>🤖 Bramha Chatbot</span>
            <span class="project-badge">LLaMA 3.3</span>
        </div>
        <div class="project-desc">Personal assistant with RAG vector search and multi-turn memory.</div>
    </div>
</div>

<div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06);">
    <h4 style="color: #34D399; font-size: 0.85rem; font-weight: 700; margin: 0 0 6px 0;">💡 System Status</h4>
    <p style="color: #9CA3AF; font-size: 0.78rem; margin: 0; line-height: 1.4;">
        RAG vector store loaded from local FAISS index. Memory keeps full context across chat turns.
    </p>
</div>
"""

footer_html = """
<div class="custom-footer">
    Designed & Developed by <b style="color: #F3F4F6;">Bramha Vinayak Gulavani</b> &nbsp;•&nbsp;
    AI & ML Engineering, <b>VIT Pune</b> &nbsp;•&nbsp;
    <a href="https://github.com/bramhagulavani/bramha-chatbot" target="_blank">View Source on GitHub ↗</a>
</div>
"""

# ── Build Gradio Interface ───────────────────────────────────────────────────
with gr.Blocks(title="🤖 Bramha's Personal AI Dashboard") as app:

    gr.HTML(header_html)

    with gr.Row():
        # Left Sidebar (scale=4)
        with gr.Column(scale=4):
            gr.HTML(sidebar_profile_html)

        # Right Chat Area (scale=8)
        with gr.Column(scale=8):
            gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(
                    height=520,
                    avatar_images=(
                        "https://api.dicebear.com/7.x/initials/svg?seed=BG&backgroundColor=10B981",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=bramha&backgroundColor=06B6D4"
                    ),
                ),
                textbox=gr.Textbox(
                    placeholder="Ask me anything about Bramha, his projects, or AI/ML skills...",
                    container=False,
                    scale=7
                ),
                examples=[
                    "Who are you?",
                    "What projects has Bramha built?",
                    "Tell me about the Plant Disease Detector",
                    "What are Bramha's AI skills & tools?",
                    "Where does Bramha study?",
                    "Give Bramha daily motivation!"
                ],
                cache_examples=False,
            )

    gr.HTML(footer_html)

# ── Launch App ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Bramha's Personal AI Dashboard...")
    app.launch(css=custom_css)