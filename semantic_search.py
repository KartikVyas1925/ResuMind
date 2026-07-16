"""
Week 7 Part A - Semantic Search with Sentence Transformers
Encodes resume sentences into embeddings and finds the most semantically
relevant sentences for a given query - even when the wording doesn't match.

Requires: pip install sentence-transformers

Run in VS Code terminal:
    python semantic_search.py
"""

import os
from sentence_transformers import SentenceTransformer, util

RESUME_FOLDER = "data/resumes"

# all-MiniLM-L6-v2: small, fast, and the standard starting model for
# sentence embeddings - same model your ResuMind project spec calls for
model = SentenceTransformer("all-MiniLM-L6-v2")


def read_resume_text(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        import fitz
        doc = fitz.open(filepath)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    return ""


def split_into_sentences(text):
    """Simple sentence splitting. Filters out very short/junk lines."""
    # Split on newlines and periods, then clean up
    raw_lines = text.replace("\n", ". ").split(". ")
    sentences = [line.strip() for line in raw_lines if len(line.strip()) > 15]
    return sentences


def build_sentence_index():
    """Reads all resumes, splits into sentences, and encodes them all."""
    all_sentences = []
    sentence_sources = []  # tracks which resume each sentence came from

    files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith((".txt", ".pdf"))]
    print(f"Found {len(files)} resumes")

    for filename in files:
        filepath = os.path.join(RESUME_FOLDER, filename)
        text = read_resume_text(filepath)
        if not text.strip():
            continue
        sentences = split_into_sentences(text)
        all_sentences.extend(sentences)
        sentence_sources.extend([filename] * len(sentences))

    print(f"Extracted {len(all_sentences)} sentences total")
    print("Encoding sentences into embeddings...")
    embeddings = model.encode(all_sentences, convert_to_tensor=True, show_progress_bar=True)

    return all_sentences, sentence_sources, embeddings


def semantic_search(query, sentences, sources, embeddings, top_k=5):
    """Finds the most semantically similar sentences to the query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} matches:\n")
    for hit in hits:
        idx = hit["corpus_id"]
        score = hit["score"]
        print(f"  [{score:.3f}] ({sources[idx]}) {sentences[idx][:150]}")


def main():
    sentences, sources, embeddings = build_sentence_index()

    # Example queries - the kind a recruiter might ask, per the roadmap's end goal
    test_queries = [
        "What experience does the candidate have with cloud technologies?",
        "Does this person have leadership or management experience?",
        "What programming languages does the candidate know?",
    ]

    for query in test_queries:
        semantic_search(query, sentences, sources, embeddings, top_k=3)
        print()


if __name__ == "__main__":
    main()