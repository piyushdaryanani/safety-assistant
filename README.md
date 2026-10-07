# Equipment Safety Assistant (RAG)

Answers equipment and safety questions from PDF manuals, shows the source page,
and refuses when the documents don't contain the answer.

## How it works
1. `ingest.py` reads every PDF in `docs/`, splits each page into passages,
   turns each passage into a vector with an embedding model, and saves them.
2. `rag.py` turns a question into a vector, finds the most similar passages
   (cosine similarity), and gives them to a local language model to write the answer.
3. `evaluate.py` runs the test questions in `questions.csv` and reports how often
   the right page was retrieved, how often the answer was correct, and how often
   it correctly refused.

## Setup (once)
    # install Ollama from https://ollama.com, then:
    ollama pull nomic-embed-text
    ollama pull llama3.2

    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install -r requirements.txt

## Run
    python3 ingest.py      # build the index (re-run after changing docs or chunk size)
    python3 ask.py         # ask questions
    python3 evaluate.py    # measure accuracy

## Settings
All settings are in `config.py`: chunk size, overlap, how many passages to
retrieve, and the minimum similarity below which it refuses.
