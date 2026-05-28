# Samsung CS RAG Demo

A Retrieval-Augmented Generation (RAG) based customer support Ωchatbot 
built on Samsung's official support documentation.

Inspired by real-world experience building Samsung's CS Chatbot service at Samsung Electronics and LLM-based NLP pipelines at NAVER Corp.

## Demo

Given a user question, the system retrieves the Top 3 most relevant 
Samsung support documents and generates a summarized answer.

**Example:**
```
Question: My Samsung phone keeps restarting by itself

📋 Related Documents:
1. Samsung phone or tablet reboots repeatedly
   🔗 https://www.samsung.com/us/support/troubleshoot/TSG10001548/

2. Different ways to restart your Galaxy phone
   🔗 https://www.samsung.com/us/support/answer/ANS10001374/

3. How to use Safe Mode
   🔗 https://www.samsung.com/us/support/answer/ANS10001997/

✨ Summary:
Random reboots are often caused by third-party apps or outdated software.
Start by checking for software updates, then boot into Safe Mode to 
identify if a downloaded app is causing the issue...
```

## Architecture

```
User Question
     ↓
Convert to Vector (all-MiniLM-L6-v2)
     ↓
FAISS Similarity Search → Top 3 Documents
     ↓
     ├── Show source documents + URLs
     └── LLM summarizes across all 3 docs (LLaMA 3.1 via Groq)
```

## Tech Stack

- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **LLM**: LLaMA 3.1 8B (via Groq API)
- **Framework**: LangChain
- **Data**: Samsung US Support pages (scraped via `scrape.py`)
- **UI**: Streamlit

## Project Structure

```
cs-rag-demo/
├── app.py              # Main RAG chatbot
├── scrape.py           # Samsung support page scraper
├── streamlit_app.py    # Streamlit UI
├── samsung_docs.jsonl  # Scraped dataset (913 pages)
├── .env.example        # Environment variable template
├── requirements.txt    # Dependencies
└── README.md
```

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/jsong951127/cs-rag-demo.git
cd cs-rag-demo
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**5. Scrape Samsung support pages (optional)**
```bash
python3 scrape.py
```

**6. Run the chatbot**
```bash
# Terminal version
python3 app.py

# Streamlit UI
streamlit run streamlit_app.py
```

## Background

At **Samsung Electronics** (2018–2023), I developed a CS Chatbot service 
deployed in the United States, including the Dialogue 
Manager and NLU components.

At **NAVER Corp** (2023–2025), I built LLM-based pipelines for the 
Knowledge Snippet service, including fine-tuning, query expansion, and 
automated validation systems.

This project combines both experiences — reimagining a CS chatbot using 
modern RAG architecture instead of the traditional BERT-based classifier approach,
with LLM-powered summarization that synthesizes answers across multiple retrieved documents.


## License
For educational purposes only.  
Samsung support content belongs to Samsung Electronics.
