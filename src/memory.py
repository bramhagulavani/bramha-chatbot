import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Personality System Prompt 
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
- Use emojis occasionally
- Never say you are ChatGPT or any other AI
"""

# ── Memory — this is the key! ────────────────────────────────────────────────
# This list stores the entire conversation history
# Every message gets added here and sent with every new request
chat_history = []

def chat_with_memory(user_message):
    # Step 1 — Add user message to history
    chat_history.append({
        "role": "user",
        "content": user_message
    })

    # Step 2 — Send system prompt + FULL history to LLaMA
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + chat_history,    # ← this is the memory magic!
        max_tokens=500
    )

    # Step 3 — Get AI response
    ai_reply = response.choices[0].message.content

    # Step 4 — Add AI response to history too
    chat_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply

# ── Test memory with a multi-turn conversation ───────────────────────────────
print("=" * 50)
print("   BRAMHA'S AI — MEMORY TEST")
print("=" * 50)

# Conversation that tests if AI remembers previous messages
conversation = [
    "Hi! My favourite programming language is Python!",
    "What is my favourite programming language?",   # should remember!
    "I am currently building a chatbot project",
    "What project am I building?",                  # should remember!
    "What two things did I just tell you about myself?"  # memory test!
]

for message in conversation:
    print(f"\n🧑 Bramha: {message}")
    reply = chat_with_memory(message)
    print(f"🤖 AI: {reply}")
    print("-" * 50)

print(f"\n📊 Total messages in memory: {len(chat_history)}")