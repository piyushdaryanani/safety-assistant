"""The same assistant built with LangChain components.

Build the index:   python3 langchain_rag.py
Evaluate it:       python3 evaluate.py --langchain

What LangChain replaces here:
  - RecursiveCharacterTextSplitter  instead of our own split()
  - OllamaEmbeddings + InMemoryVectorStore  instead of NumPy vectors and a dot product
  - ChatPromptTemplate | ChatOllama  instead of our own request to Ollama
"""
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingest import read_pages

STORE_FILE = "langchain_store.json"

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer workplace equipment and safety questions using ONLY the "
     "passages provided. If the passages do not contain the answer, reply "
     "exactly: " + config.REFUSAL + " Keep the answer short and practical."),
    ("human", "Passages:\n{context}\n\nQuestion: {question}"),
])


def embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def chat_model():
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)


def build_store():
    """Read the PDFs, split them, embed the passages, and save the store."""
    pages = []
    for name in sorted(os.listdir(config.DOCS_DIR)):
        if name.lower().endswith(".pdf"):
            for page, text in read_pages(os.path.join(config.DOCS_DIR, name)):
                pages.append(Document(page_content=text,
                                      metadata={"file": name, "page": page}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP)
    passages = splitter.split_documents(pages)      # keeps each page's metadata
    print(f"{len(passages)} passages. Embedding...")

    store = InMemoryVectorStore(embeddings())
    # Send the passages to Ollama in small batches; one huge request can crash it.
    for start in range(0, len(passages), 32):
        store.add_documents(passages[start:start + 32])
    store.dump(STORE_FILE)
    print(f"Saved {STORE_FILE}.")


def load_index():
    """Same shape as rag.load_index(), so evaluate.py can use either version."""
    return InMemoryVectorStore.load(STORE_FILE, embeddings()), None


def answer(question, store, _unused=None):
    """Return (answer_text, passages_used), same as rag.answer()."""
    found = store.similarity_search_with_score(question, k=config.TOP_K)
    hits = [{"file": d.metadata["file"], "page": d.metadata["page"],
             "text": d.page_content, "score": float(score)} for d, score in found]

    if hits[0]["score"] < config.MIN_SCORE:
        return config.REFUSAL, hits

    context = "\n\n".join(f"[{h['file']}, page {h['page']}]\n{h['text']}" for h in hits)
    chain = PROMPT | chat_model() | StrOutputParser()
    reply = chain.invoke({"context": context, "question": question}).strip()

    if config.REFUSAL.lower().rstrip(".") in reply.lower():
        return config.REFUSAL, hits
    return f"{reply}\n(Source: {hits[0]['file']}, page {hits[0]['page']})", hits


if __name__ == "__main__":
    build_store()
