"""Step 2: retrieve the best passages for a question and write an answer."""
import json

import numpy as np

import config
from ollama_client import chat, embed

SYSTEM = (
    "You answer workplace equipment and safety questions using ONLY the "
    "passages provided. If the passages do not contain the answer, reply "
    f"exactly: {config.REFUSAL} Keep the answer short and practical."
)


def load_index():
    vectors = np.load(config.INDEX_FILE)["vectors"]
    with open(config.CHUNKS_FILE) as f:
        chunks = json.load(f)
    return vectors, chunks


def retrieve(question, vectors, chunks, k=config.TOP_K):
    """Return the k passages most similar to the question, best first."""
    q = np.array(embed([question])[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vectors @ q                        # cosine similarity to every passage
    best = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


def answer(question, vectors, chunks):
    """Return (answer_text, passages_used)."""
    hits = retrieve(question, vectors, chunks)
    # Guard 1: if nothing is similar enough, don't ask the model at all.
    if hits[0]["score"] < config.MIN_SCORE:
        return config.REFUSAL, hits

    context = "\n\n".join(
        f"[{h['file']}, page {h['page']}]\n{h['text']}" for h in hits)
    reply = chat(SYSTEM, f"Passages:\n{context}\n\nQuestion: {question}")

    # Guard 2: the model itself may say the answer isn't there.
    if config.REFUSAL.lower().rstrip(".") in reply.lower():
        return config.REFUSAL, hits
    source = f"(Source: {hits[0]['file']}, page {hits[0]['page']})"
    return f"{reply}\n{source}", hits
