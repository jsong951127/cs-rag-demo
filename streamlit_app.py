import streamlit as st
import json
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Samsung CS RAG Demo",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Samsung CS RAG Chatbot Demo")
st.caption("Ask any question about Samsung device troubleshooting")

@st.cache_resource
def load_vectorstore():
    with st.spinner("Loading Samsung support documents..."):
        docs = []
        with open("samsung_docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        # chunk by subtitles
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")]
        )

        chunks = []
        for doc in docs:
            sections = splitter.split_text(doc["content"])
            for section in sections:
                section.metadata["url"] = doc["url"]
                section.metadata["title"] = doc["title"]
                section.page_content = f"{doc['title']}\n{section.page_content}"
                chunks.append(section)

        # Creating Vector
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

vectorstore = load_vectorstore()

# LLM for summarization
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a Samsung customer support assistant.
Answer the user's question based on the documents below.
If the answer is not in the documents, say "I couldn't find relevant information. Please contact Samsung Support directly."

Documents:
{context}

Question: {question}

Provide a clear and concise summary answer based on all the documents above.
""")

chain = prompt | llm | StrOutputParser()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about your Samsung device..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # Retrieve Top 3 documents
        top_docs = vectorstore.similarity_search(question, k=3)

        with st.expander("📋 Related Documents"):
            for i, doc in enumerate(top_docs):
                title = doc.metadata.get("title", "Unknown")
                url = doc.metadata.get("url", "")
                content_without_title = "\n".join(doc.page_content.split("\n")[1:])
                st.markdown(f"**{i+1}. {title}**")
                st.markdown(f"🔗 [{url}]({url})")
                st.caption(f"{content_without_title}")
                if i < 2:
                    st.divider()

        # Summary
        context_parts = []
        for i, doc in enumerate(top_docs):
            title = doc.metadata.get("title", "")
            context_parts.append(f"[Document {i+1}] {title}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        with st.spinner("Generating answer..."):
            answer = chain.invoke({"context": context, "question": question})

        st.markdown("**✨ Summary Answer:**")
        st.markdown(answer)

        full_response = f"**✨ Summary Answer:**\n{answer}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})