"""Step 3: measure how good the assistant is.

questions.csv has one row per test question:
  question     the question to ask
  file         the PDF that contains the answer (empty if it is not in any document)
  page         the PDF page number of the answer (empty if not in any document);
               if the fact is on several pages, list them like 17|21
  keywords     words the answer must contain, separated by ';'
               (use | between acceptable alternatives, like 50|1270)

Run:  python3 evaluate.py              (plain Python version, saves results.csv)
      python3 evaluate.py --langchain  (LangChain version, saves results_langchain.csv)
"""
import csv
import sys

import config

# "python3 evaluate.py --langchain" tests the LangChain version instead.
USE_LANGCHAIN = "--langchain" in sys.argv
if USE_LANGCHAIN:
    from langchain_rag import answer, load_index
else:
    from rag import answer, load_index


def has_keywords(text, keywords):
    """True if the answer contains every required keyword.
    Each keyword may list alternatives separated by '|'."""
    text = text.lower()
    for group in keywords.lower().split(";"):
        options = [o.strip() for o in group.split("|") if o.strip()]
        if options and not any(o in text for o in options):
            return False
    return True


def main():
    vectors, chunks = load_index()
    with open("questions.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    retrieved_ok = answered_ok = answerable = 0
    refused_ok = unanswerable = 0
    results = []

    for row in rows:
        text, hits = answer(row["question"], vectors, chunks)
        refused = text == config.REFUSAL

        if row["file"]:                                   # answer is in the documents
            answerable += 1
            pages = row["page"].split("|")
            found = any(h["file"] == row["file"] and str(h["page"]) in pages
                        for h in hits)
            correct = (not refused) and has_keywords(text, row["keywords"])
            retrieved_ok += found
            answered_ok += correct
            results.append([row["question"], "yes" if found else "no",
                            "yes" if correct else "no", text])
        else:                                             # it should refuse
            unanswerable += 1
            refused_ok += refused
            results.append([row["question"], "-", "yes" if refused else "no", text])

    out_file = "results_langchain.csv" if USE_LANGCHAIN else "results.csv"
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "right_page_retrieved", "correct", "answer"])
        w.writerows(results)

    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    print(f"Settings: chunk size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}, "
          f"top {config.TOP_K}, minimum score {config.MIN_SCORE}")
    print(f"Right page retrieved:      {pct(retrieved_ok, answerable)}")
    print(f"Answer correct:            {pct(answered_ok, answerable)}")
    print(f"Refused when not in docs:  {pct(refused_ok, unanswerable)}")
    print(f"Details saved to {out_file}")


if __name__ == "__main__":
    main()
