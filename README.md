# 💬 OpenAI RAG Chat With GitHub Topic | Build RAG Index & Chat UI for GitHub Topic Repositories

<p align="center">
  <img src="https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge" alt="OpenAI API">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge" alt="Streamlit UI">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge" alt="Language Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT">
</p>

**Create a robust Retrieval-Augmented Generation (RAG) index of large GitHub repositories filtered by topic, and interactively chat with them using OpenAI's file search and a Streamlit interface.**

## 📑 Table of Contents
- [Overview](#-overview)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Issues & Support](#-issues--support)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Overview

This prototype contains scripts to pull GitHub repositories by topic, render them as HTML, index them using OpenAI's file search (RAG), and interact with them via a Streamlit chat interface.

```bash
$ python rendergit-topic.py machine-learning
Rendering repository: tensorflow/tensorflow ...
Rendering repository: pytorch/pytorch ...
$ python openai-file-search.py
Creating RAG index for 2 repositories ...
```

- `rendergit-topic.py`: Takes a GitHub Topic as a CLI argument and calls `rendergit` on each repo - turning each repo into a potentially large (>100MB) HTML file.
- `openai-file-search.py`: Creates a RAG index of all of these files using OpenAI. This can take a substantial amount of time for a large topic.
- `rag-chat-multi.py`: A familiar Streamlit chatbot UI. Automatically connects to an OpenAI file search (RAG) ID if available. Supports file uploads, specification of the API key and model selection. Has some quirks.

## 💻 Installation & Setup

- Python 3.x
- OpenAI API Key
- `rendergit` tool
- Required Python libraries: `streamlit`, `openai` (and any other dependencies used in the scripts).

> [!NOTE]
> Ensure you have an active OpenAI API key with access to file search (Assistants API).

## 💡 Usage

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

## 🐛 Issues & Support

If you encounter any bugs, have feature requests, or need assistance troubleshooting your RAG setup, please [open an issue](../../issues) in this repository. 

## 🤝 Contributing

Contributions are always welcome! Whether it's adding new models, optimizing the Streamlit UI, or improving repository parsing, your help is appreciated. 
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
