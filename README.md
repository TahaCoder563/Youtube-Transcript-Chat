# 🎥 YouTube Transcript Chat App

A Streamlit-based web application that lets users ask questions about the content of a YouTube video using its English transcript. It extracts the transcript, breaks it into chunks, embeds it using HuggingFace models, and answers questions via the Perplexity LLM.

## 🚀 Features

- 🔗 Enter any YouTube video URL with English captions
- 🤖 Ask natural language questions about the video
- 🧠 Powered by Perplexity LLM & HuggingFace sentence embeddings
- 📚 Context-aware QA using FAISS vector store
- 🎨 Clean UI with custom background and styled answer box

The core Streamlit app that:
- Accepts a YouTube URL and user question
- Fetches and processes transcript using `youtube-transcript-api`
- Splits transcript into chunks with `RecursiveCharacterTextSplitter`
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in FAISS and retrieves top `k` similar chunks
- Feeds context and question into a prompt chain using Perplexity
- Displays the generated answer in a styled output box

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **LangChain**
- **HuggingFace Sentence Transformers**
- **Perplexity Chat API**
- **FAISS**
- **YouTube Transcript API**

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TahaCoder563/Youtube-Transcript-Chat.git
   cd Youtube-Transcript-Chat
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Set up API keys**
   
   **🔑 Get a Perplexity API Key:**
    * Go to https://docs.perplexity.ai
    *  Sign in or create an account
    *  Generate a new API key from the dashboard

   **🔑 Get a HuggingFace API Token:**
    * Go to https://huggingface.co/settings/tokens
    * Create a token with read access
   
   Now paste your keys in the .env file:
  
4. **Run the app**
   ```bash
   streamlit run main.py
