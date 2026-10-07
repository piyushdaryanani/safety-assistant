"""The two calls we make to Ollama: embed text, and chat."""
import requests
import config


def embed(texts):
    """Return one vector (list of floats) per input string."""
    vectors = []
    for start in range(0, len(texts), 32):          # small batches
        batch = texts[start:start + 32]
        r = requests.post(f"{config.OLLAMA_URL}/api/embed",
                          json={"model": config.EMBED_MODEL, "input": batch},
                          timeout=300)
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    return vectors


def chat(system, user):
    """Ask the chat model one question and return its reply as text."""
    r = requests.post(f"{config.OLLAMA_URL}/api/chat",
                      json={"model": config.CHAT_MODEL, "stream": False,
                            "options": {"temperature": 0},
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]},
                      timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
