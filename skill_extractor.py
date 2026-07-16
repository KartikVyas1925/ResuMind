"""
Week 2 - Rule-Based Skill & Title Extractor
Uses spaCy's Matcher and PhraseMatcher to extract job titles and skills
from resume text WITHOUT any machine learning - pure pattern matching.

Run in VS Code terminal:
    python skill_extractor.py
"""

import os
import csv
import spacy
from spacy.matcher import Matcher, PhraseMatcher

# ---- CONFIG ----
RESUME_FOLDER = "data/resumes"
OUTPUT_CSV = "outputs/extracted_skills.csv"

nlp = spacy.load("en_core_web_sm")

# ---- 1. PhraseMatcher for skills ----
# A predefined list of skills to look for. Expand this list as needed -
# this is the "high-precision, low-recall" approach from the roadmap:
# it finds exactly what's in this list, nothing more, nothing less.
SKILL_LIST = [
    "Python", "SQL", "Java", "JavaScript", "C++", "Machine Learning",
    "Deep Learning", "Project Management", "Data Analysis", "Excel",
    "Tableau", "Power BI", "AWS", "Azure", "Docker", "Kubernetes",
    "Git", "Agile", "Scrum", "Leadership", "Communication",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "React", "Node.js", "HTML", "CSS", "Marketing", "SEO",
    "Content Strategy", "Salesforce", "CRM", "Six Sigma",
]

skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # case-insensitive matching
skill_patterns = [nlp.make_doc(skill) for skill in SKILL_LIST]
skill_matcher.add("SKILL", skill_patterns)


# ---- 2. Matcher for job title patterns ----
# Rule-based patterns (not exact phrases) - e.g. catches "Chief Executive Officer"
# OR "CEO" using token-level rules instead of one fixed string.
title_matcher = Matcher(nlp.vocab)

ceo_pattern = [{"LOWER": "chief"}, {"LOWER": "executive"}, {"LOWER": "officer"}]
ceo_short_pattern = [{"TEXT": "CEO"}]
manager_pattern = [{"LOWER": {"IN": ["senior", "junior", "lead"]}, "OP": "?"},
                    {"POS": "NOUN", "OP": "*"},
                    {"LOWER": "manager"}]
engineer_pattern = [{"LOWER": {"IN": ["senior", "junior", "lead"]}, "OP": "?"},
                     {"POS": "NOUN", "OP": "*"},
                     {"LOWER": "engineer"}]

title_matcher.add("JOB_TITLE", [ceo_pattern, ceo_short_pattern, manager_pattern, engineer_pattern])


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


def extract_skills_and_titles(text, resume_id):
    doc = nlp(text)
    rows = []

    # Run skill matcher
    skill_matches = skill_matcher(doc)
    seen_skills = set()
    for match_id, start, end in skill_matches:
        span = doc[start:end]
        skill_text = span.text
        if skill_text.lower() not in seen_skills:  # avoid duplicate rows for repeated mentions
            seen_skills.add(skill_text.lower())
            rows.append({
                "resume_id": resume_id,
                "match_type": "SKILL",
                "matched_text": skill_text,
            })

    # Run title matcher
    title_matches = title_matcher(doc)
    seen_titles = set()
    for match_id, start, end in title_matches:
        span = doc[start:end]
        title_text = span.text
        if title_text.lower() not in seen_titles:
            seen_titles.add(title_text.lower())
            rows.append({
                "resume_id": resume_id,
                "match_type": "JOB_TITLE",
                "matched_text": title_text,
            })

    return rows


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_rows = []
    files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith((".txt", ".pdf"))]
    print(f"Found {len(files)} resume files")

    for filename in files:
        filepath = os.path.join(RESUME_FOLDER, filename)
        resume_id = os.path.splitext(filename)[0]

        text = read_resume_text(filepath)
        if not text.strip():
            continue

        rows = extract_skills_and_titles(text, resume_id)
        all_rows.extend(rows)
        skill_count = sum(1 for r in rows if r["match_type"] == "SKILL")
        title_count = sum(1 for r in rows if r["match_type"] == "JOB_TITLE")
        print(f"  {filename}: {skill_count} skills, {title_count} titles found")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "match_type", "matched_text"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} matches to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()