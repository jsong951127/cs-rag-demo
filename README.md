# Samsung CS RAG Demo

A Retrieval-Augmented Generation (RAG) based customer support chatbot
built on Samsung's official support documentation.

Inspired by real-world experience building Samsung's CS Chatbot service at Samsung Electronics and LLM-based NLP pipelines at NAVER Corp.

## Demo

Link: *https://jamie-cs-rag-demo.streamlit.app/*

Given a user question, the system retrieves the Top 3 most relevant 
Samsung support documents and generates a summarized answer.

**Example:**
```
Question: My phone gets very hot while charging

📋 Related Documents:
1. Keep your Galaxy device at its normal operating temperature
   🔗 https://www.samsung.com/us/support/answer/ANS10002887/
   📄 Here are some tips for what to do if your Galaxy devices warms up:
Disconnect the charger, and close any running apps until the device cools down. If you are wearing earbuds, stop playing music, don't...

2. Keep your Galaxy device at its normal operating temperature
   🔗 https://www.samsung.com/us/support/answer/ANS10002887/
   📄 Although your Galaxy devices may get warm sometimes, they have built-in safeguards to alert you and protect themselves if their temperature exceeds the normal operating range. Check out the CTIA's use...

3. Samsung phone or tablet will not charge
   🔗 https://www.samsung.com/us/support/troubleshoot/TSG10001462/
   📄 If your device is reporting that the temperature is too low to charge, unplug the charger and allow it to return to a normal temperature before proceeding.
Verify that the device, charger, and USB cab...

✨ Summary:
Based on the provided documents, if your phone gets very hot while charging, here are some steps you can take:

1. Disconnect the charger and close any running apps until the device cools down.
2. Try using a different Samsung-approved charger (and USB cable) when charging.
3. Remove metal or magnetic material between the device and the charger if using a wireless charger.
...

If the issue persists, it's recommended to contact Samsung support directly to further assist you.
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
├── samsung_docs.jsonl  # Scraped dataset (909 pages)
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
