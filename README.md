# 💬 OpenAI RAG Chat With GitHub Topic

<p align="center">
  <img src="https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge" alt="OpenAI API">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge" alt="Streamlit UI">
  <img src="https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge" alt="Python 3">
</p>

<p align="center"><b>Create a RAG index of GitHub repos by topic and chat with them using OpenAI and Streamlit.</b></p>

---

## 📑 Index
- [Overview](#-overview)
- [Quickstart](#-quickstart)
- [Prerequisites & Dependencies](#-prerequisites--dependencies)

---

## 🚀 Overview

This prototype contains scripts to pull GitHub repositories by topic, render them as HTML, index them using OpenAI's file search (RAG), and interact with them via a Streamlit chat interface.

- `rendergit-topic.py`: Takes a GitHub Topic as a CLI argument and calls `rendergit` on each repo - turning each repo into a potentially large (>100MB) HTML file.
- `openai-file-search.py`: Creates a RAG index of all of these files using OpenAI. This can take a substantial amount of time for a large topic.
- `rag-chat-multi.py`: A familiar Streamlit chatbot UI. Automatically connects to an OpenAI file search (RAG) ID if available. Supports file uploads, specification of the API key and model selection. Has some quirks.

---

## 🏁 Quickstart

To use this project, run the scripts in the following order:

1. Render repositories by topic:
   ```bash
   python rendergit-topic.py <topic>
   ```
2. Create the RAG index:
   ```bash
   python openai-file-search.py
   ```
3. Start the Streamlit chat UI:
   ```bash
   streamlit run rag-chat-multi.py
   ```

---

## 📦 Prerequisites & Dependencies

- Python 3.x
- OpenAI API Key
- `rendergit` tool
- Required Python libraries: `streamlit`, `openai` (and any other dependencies used in the scripts).

---
