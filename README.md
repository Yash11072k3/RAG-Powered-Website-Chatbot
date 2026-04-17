# 🚀 RAG Powered Website Chatbot

## 📌 Overview

The **RAG Powered Website Chatbot** is an AI-based application that allows users to input any website URL and ask questions based strictly on its content.

This project uses **Retrieval-Augmented Generation (RAG)** to extract, process, and retrieve relevant information from a website, ensuring that answers are accurate and grounded only in the provided content.

Unlike traditional chatbots, this system **does not rely on general knowledge** and avoids hallucinations by strictly using website data.

---

## 🎯 Key Features

* 🌐 Scrapes both **static and dynamic websites**
* 🧹 Cleans HTML by removing navigation, ads, and unnecessary elements
* ✂️ Splits text into optimized chunks with overlap
* 🧠 Generates embeddings using Sentence Transformers
* 🔍 Implements **Hybrid Search (Semantic + Keyword)**
* ⚡ Fast retrieval using FAISS vector database
* 🤖 Uses local LLM via Ollama (Phi model)
* 🚫 Strict answer control (no hallucinations)
* 💬 Interactive chat interface using Streamlit
* 🔎 Displays source previews for transparency

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **FAISS (Vector Database)**
* **Sentence Transformers (all-MiniLM-L6-v2)**
* **BeautifulSoup**
* **Playwright**
* **Requests**
* **Ollama (Local LLM - Phi model)**

---

## 🏗️ System Architecture

### Step 1: Website Input

User enters a website URL in the Streamlit interface.

### Step 2: Web Scraping

* Uses **Playwright** for dynamic websites
* Falls back to **Requests** for static pages
* Extracts meaningful content only

### Step 3: Data Cleaning

* Removes scripts, navigation bars, footers, and noise
* Keeps only main content

### Step 4: Text Chunking

* Splits content into chunks (with overlap)
* Ensures better context understanding

### Step 5: Embedding Generation

* Converts text chunks into vector embeddings
* Uses Sentence Transformers

### Step 6: Vector Storage

* Stores embeddings in **FAISS index**
* Enables fast similarity search

### Step 7: Hybrid Retrieval (Core Innovation)

* Combines:

  * Semantic similarity (vector search)
  * Keyword matching (lexical scoring)
* Improves accuracy significantly

### Step 8: Answer Generation

* Retrieves most relevant chunks
* Sends them to **Ollama (Phi model)**
* Generates answer strictly from context

---

## 📂 Project Structure

```
rag-powered-website-chatbot/
│── app.py          # Main Streamlit UI and chatbot logic
│── scraper.py      # Website scraping and HTML cleaning
│── embeddings.py   # Text processing, embeddings, hybrid search
│── llm.py          # LLM integration with strict prompt control
│── README.md
│── requirements.txt
│── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-powered-website-chatbot.git
cd rag-powered-website-chatbot
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
playwright install
```

---

### 3. Install and Run Ollama

Download from: https://ollama.com

Run the model:

```bash
ollama run phi
```

---

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📖 Usage

1. Enter a website URL
2. Click **"Load Website"**
3. Wait for scraping and indexing
4. Ask questions in chat
5. View answers along with source previews

---

## 🧠 Approach

This project uses a **Retrieval-Augmented Generation (RAG)** pipeline:

* Instead of sending the entire website to the model,
* It retrieves only the most relevant chunks,
* Then generates answers using those chunks.

### Why this approach?

* Improves accuracy ✅
* Reduces hallucination ✅
* Ensures context-based answers ✅

---

## 🚫 Strict Answer Policy

The chatbot follows strict rules:

* Answers only from website content
* Does not use external knowledge
* Does not guess
* If answer is not found, returns:

**"This information is not in the website."**

---

## ⚠️ Challenges Faced

* Handling **403 Forbidden errors**
* Extracting content from **dynamic websites**
* Avoiding hallucinated answers from LLM
* Balancing semantic and keyword search
* Optimizing performance for local systems

---

## 🔮 Future Improvements

* Multi-page website crawling
* Chat memory support
* Better UI/UX design
* Support for PDFs and documents
* Cloud deployment (AWS / Docker)

---

## 📌 Conclusion

This project demonstrates a **complete end-to-end RAG pipeline using a local LLM**, combining:

* Web scraping
* Natural Language Processing
* Vector search
* AI-powered question answering

It showcases practical skills in **AI Engineering and real-world application development**.

---

## 👤 Author

**Yashwanth S**

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
