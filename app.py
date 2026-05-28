from dotenv import load_dotenv
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Load Documents
print("Loading Documents...")
docs = []
with open("samsung_docs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))
print(f"{len(docs)} documents loaded.")

# 2. ## chunk by sections
print("Chunking...")
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

print(f"{len(chunks)} chunks generated.")

# 3. Creating Vector
print("Creating Vector...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Storing Vector...")

# 4. LLM
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

# 5. 검색 + 답변
print("\n=== Samsung CS RAG Chatbot ===\n")

while True:
    question = input("Question (q to quit): ").strip()
    if question == "q":
        break

    # Top 3 문서 검색
    top_docs = vectorstore.similarity_search(question, k=3)

    # 관련 문서 출력
    print("\n📋 Relevant Documents:")
    seen_urls = []
    context_parts = []

    for i, doc in enumerate(top_docs):
        title = doc.metadata.get("title", "Unknown")
        url = doc.metadata.get("url", "")
        content_preview = doc.page_content[:200].replace("\n", " ")

        print(f"\n{i + 1}. {title}")
        print(f"   🔗 {url}")
        print(f"   📄 {content_preview}...")

        context_parts.append(f"[Document {i + 1}] {title}\n{doc.page_content}")
        seen_urls.append(url)

    # 종합 요약 답변
    context = "\n\n".join(context_parts)
    answer = chain.invoke({"context": context, "question": question})

    print(f"\n✨ 요약 답변:\n{answer}\n")
    print("-" * 50)