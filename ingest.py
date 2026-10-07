"""Step 1: read the PDFs, split them into passages, and build the index.

Run:  python3 ingest.py
"""
import json
import os

import numpy as np
from pypdf import PdfReader

import config
from ollama_client import embed


def read_pages(path):
    """Yield (page_number, text) for every page that has text."""
    reader = PdfReader(path)
    for number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())   # tidy whitespace
        if text:
            yield number, text


def split(text, size, overlap):
    """Cut text into passages of about `size` characters that overlap a little,
    so a sentence cut at a boundary still appears whole in one passage."""
    passages = []
    start = 0
    while start < len(text):
        passages.append(text[start:start + size])
        start += size - overlap
    return passages


def build_chunks():
    chunks = []
    for name in sorted(os.listdir(config.DOCS_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        for page, text in read_pages(os.path.join(config.DOCS_DIR, name)):
            for passage in split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append({"file": name, "page": page, "text": passage})
    return chunks


def main():
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"No PDF text found in '{config.DOCS_DIR}/'. Add manuals first.")
    print(f"{len(chunks)} passages from "
          f"{len({c['file'] for c in chunks})} documents. Embedding...")

    vectors = np.array(embed([c["text"] for c in chunks]), dtype=np.float32)
    # Normalise each vector to length 1 so a dot product equals cosine similarity.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    np.savez_compressed(config.INDEX_FILE, vectors=vectors)
    with open(config.CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    print(f"Saved {config.INDEX_FILE} and {config.CHUNKS_FILE}.")


if __name__ == "__main__":
    main()
