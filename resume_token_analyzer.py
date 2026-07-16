"""
Week 1 - Resume Token Analyzer
Reads resumes from a folder, runs spaCy NLP pipeline, and exports
tokens, lemmas, POS tags, and named entities to a CSV.

Run in VS Code terminal:
    python resume_token_analyzer.py
"""

import os
import csv
import spacy

# ---- CONFIG - change these paths to match your project ----
RESUME_FOLDER = "data/resumes"          # folder containing your resume .txt or .pdf files
OUTPUT_CSV = "outputs/resume_tokens.csv"

nlp = spacy.load("en_core_web_sm")


def read_resume_text(filepath):
    """Reads a resume file. Handles .txt directly; for .pdf, uses PyMuPDF."""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif filepath.endswith(".pdf"):
        import fitz  # PyMuPDF - install with: pip install pymupdf
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    else:
        return ""


def analyze_resume(text, resume_id):
    """Runs spaCy on the text and returns a list of rows (one per token)."""
    doc = nlp(text)
    rows = []
    for token in doc:
        if token.is_space:
            continue
        rows.append({
            "resume_id": resume_id,
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "is_stop": token.is_stop,
        })
    return rows


def extract_entities(text, resume_id):
    """Extracts named entities (people, orgs, dates, etc.) from the text."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "resume_id": resume_id,
            "entity_text": ent.text,
            "entity_label": ent.label_,
        })
    return entities


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_token_rows = []
    all_entity_rows = []

    files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith((".txt", ".pdf"))]
    print(f"Found {len(files)} resume files in {RESUME_FOLDER}")

    for filename in files:
        filepath = os.path.join(RESUME_FOLDER, filename)
        resume_id = os.path.splitext(filename)[0]

        text = read_resume_text(filepath)
        if not text.strip():
            print(f"  Skipping {filename} - no text extracted")
            continue

        token_rows = analyze_resume(text, resume_id)
        entity_rows = extract_entities(text, resume_id)

        all_token_rows.extend(token_rows)
        all_entity_rows.extend(entity_rows)
        print(f"  Processed {filename}: {len(token_rows)} tokens, {len(entity_rows)} entities")

    # Write tokens CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "token", "lemma", "pos", "dep", "is_stop"])
        writer.writeheader()
        writer.writerows(all_token_rows)
    print(f"\nSaved token analysis to {OUTPUT_CSV}")

    # Write entities CSV (separate file - useful for later weeks too)
    entities_csv = OUTPUT_CSV.replace(".csv", "_entities.csv")
    with open(entities_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "entity_text", "entity_label"])
        writer.writeheader()
        writer.writerows(all_entity_rows)
    print(f"Saved entity analysis to {entities_csv}")


if __name__ == "__main__":
    main()