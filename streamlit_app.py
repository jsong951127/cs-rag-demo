import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Samsung CS Chatbot",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Samsung CS RAG Chatbot")
st.caption("Ask any question about Samsung device troubleshooting")

# 벡터스토어 캐싱 (매번 다시 만들지 않게)
@st.cache_resource
def load_vectorstore():
    with st.spinner("Loading Samsung support documents..."):
        docs = []
        with open("samsung_docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")]
        )

        chunks = []
        for doc in docs:
            sections = splitter.split_text(doc["content"])
            for section in sections:
                section.metadata["url"] = doc["url"]
                section.metadata["title"] = doc["title"]
                chunks.append(section)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

vectorstore = load_vectorstore()

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

# 채팅 히스토리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 입력창
if question := st.chat_input("Ask about your Samsung device..."):

    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 답변 생성
    with st.chat_message("assistant"):
        # Top 3 문서 검색
        top_docs = vectorstore.similarity_search(question, k=3)

        # 관련 문서 표시
        with st.expander("📋 Related Documents"):
            for i, doc in enumerate(top_docs):
                title = doc.metadata.get("title", "Unknown")
                url = doc.metadata.get("url", "")
                preview = doc.page_content[:200].replace("\n", " ")
                st.markdown(f"**{i+1}. {title}**")
                st.markdown(f"🔗 [{url}]({url})")
                st.caption(f"{preview}...")
                if i < 2:
                    st.divider()

        # 종합 요약 답변
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