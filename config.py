"""Settings for the whole project, in one place."""

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # turns text into vectors
CHAT_MODEL = "llama3.2"            # writes the final answer

DOCS_DIR = "docs"                  # put the PDF manuals here
INDEX_FILE = "index.npz"           # vectors, one row per passage
CHUNKS_FILE = "chunks.json"        # the passage text and where it came from

CHUNK_SIZE = 600                  # characters per passage
CHUNK_OVERLAP = 100                # characters shared by neighbouring passages
TOP_K = 4                          # passages given to the model
MIN_SCORE = 0.55                   # below this similarity, refuse to answer

REFUSAL = "I can't find that in the documents."
