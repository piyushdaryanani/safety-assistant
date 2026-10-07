"""Ask questions from the terminal.   Run:  python3 ask.py"""
from rag import answer, load_index


def main():
    vectors, chunks = load_index()
    print("Ask a question (or press Enter to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        text, _ = answer(question, vectors, chunks)
        print(text)


if __name__ == "__main__":
    main()
