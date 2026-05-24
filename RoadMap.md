# 🤖 Bramha Chatbot AI — Project Roadmap

> A personal AI chatbot powered by LLaMA 3 and Groq — with memory, personality and RAG.
> Built by **Bramha Vinayak Gulavani** | AI & ML Student, VIT Pune

---

## 🧠 Tech Stack

| Tool | Purpose |
|:-----|:--------|
| LLaMA 3 | Core AI language model (by Meta) |
| Groq API | Free, blazing fast LLaMA 3 gateway |
| LangChain | Conversation memory and RAG pipeline |
| FAISS | Vector database for personal knowledge |
| Gradio | WhatsApp-style chat web interface |
| Python-dotenv | Secure API key management |

---

## 📍 Phase Overview

```
Phase 0 → Setup
Phase 1 → Connect to LLaMA 3
Phase 2 → Personality Training
Phase 3 → Conversation Memory
Phase 4 → Personal Knowledge Base (RAG)
Phase 5 → WhatsApp-style Web App
Phase 6 → Deploy Publicly
```

---

## ✅ Phase 0 — Setup `Day 1`

**Goal:** Project folder, GitHub repo, venv, Groq API key

### Tasks
- [x] Create folder: `C:\Projects\bramha-chatbot`
- [x] Initialize GitHub repo: `bramha-chatbot`
- [x] Set up Python virtual environment
- [x] Create `.gitignore` to protect API keys
- [x] Create project folder structure
- [ ] Get free Groq API key at console.groq.com
- [ ] Store API key in `.env` file
- [ ] Install all required libraries
- [ ] Verify all libraries load correctly

### Libraries to install
```bash
pip install groq langchain langchain-groq gradio python-dotenv faiss-cpu langchain-community sentence-transformers
```

### Folder structure
```
bramha-chatbot/
├── src/          ← app.py goes here
├── data/         ← personal knowledge files
├── models/       ← saved embeddings
├── .env          ← API keys (never pushed to GitHub!)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⏳ Phase 1 — Connect to LLaMA 3 `Day 2`

**Goal:** Send your first message to LLaMA 3 and get a response back

### Tasks
- [ ] Store Groq API key safely using python-dotenv
- [ ] Connect to Groq API using Python
- [ ] Send a test message to LLaMA 3
- [ ] Print the response in terminal
- [ ] Test with multiple questions

### What you'll learn
- How to use API keys safely
- How LLM API calls work
- Request → AI thinks → Response flow

> **Simple explanation:** Groq is like a superfast free gateway to LLaMA 3. You send a message → LLaMA 3 thinks → response comes back in under 1 second!

---

## ⏳ Phase 2 — Personality Training `Days 3–4`

**Goal:** Give the chatbot YOUR personality using a system prompt

### Tasks
- [ ] Write a detailed system prompt about yourself
- [ ] Add your name, college, interests and goals
- [ ] Add your preferred tone and response style
- [ ] Add knowledge about your projects — Plant Disease AI, Pune Rent Predictor
- [ ] Test personality with various questions
- [ ] Refine the system prompt based on responses

### What you'll learn
- What a system prompt is and how it works
- How to control AI behavior through prompting
- Prompt engineering fundamentals

> **Simple explanation:** The system prompt is like giving the AI a detailed briefing about who it is and how it should behave — just like a job description! This is how you "train" its personality without touching any model weights.

---

## ⏳ Phase 3 — Conversation Memory `Days 5–6`

**Goal:** Make the chatbot remember what was said earlier in the conversation

### Tasks
- [ ] Implement chat history list using LangChain
- [ ] Pass full conversation history with every new message
- [ ] Test multi-turn conversations
- [ ] Add memory limit to avoid token overflow
- [ ] Handle memory clearing between sessions

### What you'll learn
- How LLMs handle context windows
- What token limits are
- How LangChain manages conversation state

> **Simple explanation:** Without memory, every message is sent fresh — the AI forgets everything instantly. With memory, you send the full conversation history each time so it remembers what you said 10 messages ago!

---

## ⏳ Phase 4 — Personal Knowledge Base `Days 7–8`

**Goal:** Feed the chatbot YOUR own documents, notes and knowledge using RAG

### Tasks
- [ ] Write a personal knowledge text file about yourself
- [ ] Add your projects, skills, college info, daily schedule
- [ ] Implement RAG using LangChain and FAISS
- [ ] Create vector embeddings of your documents
- [ ] Connect retriever to chatbot pipeline
- [ ] Test — ask "what are my projects?" and see it read YOUR notes!

### What you'll learn
- What RAG (Retrieval Augmented Generation) is
- How vector embeddings work
- How FAISS stores and searches knowledge
- End-to-end RAG pipeline

> **Simple explanation:** RAG means the chatbot searches your personal documents before answering. Ask "what are my projects?" and it actually reads YOUR notes to answer — not just guessing from its training data!

---

## ⏳ Phase 5 — WhatsApp-style Web App `Days 9–10`

**Goal:** Build a beautiful chat interface in the browser

### Tasks
- [ ] Build chat UI using Gradio ChatInterface
- [ ] Add message bubbles like WhatsApp
- [ ] Add typing indicator while AI is thinking
- [ ] Add clear chat / new conversation button
- [ ] Dark themed beautiful UI
- [ ] Connect all phases — personality + memory + RAG + UI

### What you'll learn
- How to build chat UIs with Gradio
- How to connect backend AI to frontend UI
- State management in chat applications

> **Simple explanation:** You open the browser, type a message, and your own personal AI chatbot replies in real time — just like WhatsApp. But this one knows YOUR personality, YOUR projects, and remembers YOUR conversation!

---

## ⏳ Phase 6 — Deploy Publicly `Day 11`

**Goal:** Share your chatbot with the world via a free public URL

### Tasks
- [ ] Create free Hugging Face account
- [ ] Create a new Space on Hugging Face
- [ ] Push project to Hugging Face Spaces
- [ ] Set API key as a secret environment variable
- [ ] Test the live public URL
- [ ] Write professional README.md
- [ ] Close all GitHub issues
- [ ] Final git commit — v1.0 shipped!

### What you'll learn
- How to deploy AI apps for free
- How to manage secrets in production
- How Hugging Face Spaces works

> **Simple explanation:** Hugging Face Spaces gives you a free public URL — share it with friends, family, professors. They open the link and chat with your personal AI. That's a fully deployed chatbot!

---

## 📊 Progress Tracker

| Phase | Description | Status | Days |
|:-----:|:------------|:------:|:----:|
| 0 | Setup + API key | ✅ In Progress | Day 1 |
| 1 | Connect to LLaMA 3 | ⏳ Pending | Day 2 |
| 2 | Personality training | ⏳ Pending | Days 3–4 |
| 3 | Conversation memory | ⏳ Pending | Days 5–6 |
| 4 | Personal knowledge base | ⏳ Pending | Days 7–8 |
| 5 | WhatsApp-style web app | ⏳ Pending | Days 9–10 |
| 6 | Deploy publicly | ⏳ Pending | Day 11 |

---

## 👨‍💻 About

Built by **Bramha Vinayak Gulavani**
Second Year AI & ML Student — VIT Pune

Previous projects:
- 🌿 [Plant Disease Detector](https://github.com/bramhagulavani/plant-disease-ai) — MobileNetV2, 94.16% accuracy
- 🏠 [Pune Rent Predictor](https://github.com/bramhagulavani/pune-rent-predictor) — XGBoost, 74.79% R²

[![GitHub](https://img.shields.io/badge/GitHub-bramhagulavani-181717?style=for-the-badge&logo=github)](https://github.com/bramhagulavani)