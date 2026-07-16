"""
Week 4 - Word Vectors & Fuzzy Skill Matching
Uses spaCy's medium model (en_core_web_md) which has real pretrained word
vectors, to compare document similarity and build a "fuzzy" skill matcher
that catches skills even when the exact wording differs.

Requires: python -m spacy download en_core_web_md

Run in VS Code terminal:
    python similarity_matcher.py
"""

import spacy

# Medium model has real word vectors - small model does not
nlp = spacy.load("en_core_web_md")


# ---------------------------------------------------------
# Part 1: Document similarity - doc.similarity()
# ---------------------------------------------------------
def compare_documents():
    print("=== Document Similarity Demo ===\n")

    doc1 = nlp("Managed a team of 5 engineers to deliver projects on time.")
    doc2 = nlp("Led a group of developers to complete deliverables punctually.")
    doc3 = nlp("Enjoys hiking and photography on weekends.")

    print(f"Doc1: {doc1.text}")
    print(f"Doc2: {doc2.text}")
    print(f"Doc3: {doc3.text}\n")

    print(f"Similarity (Doc1 vs Doc2 - similar meaning, different words): {doc1.similarity(doc2):.3f}")
    print(f"Similarity (Doc1 vs Doc3 - unrelated content): {doc1.similarity(doc3):.3f}")
    print("\nNotice: Doc1 vs Doc2 should score noticeably higher than Doc1 vs Doc3,")
    print("even though Doc1 and Doc2 share almost no identical words.\n")


# ---------------------------------------------------------
# Part 2: Fuzzy skill matcher using cosine similarity
# ---------------------------------------------------------
# Standard/canonical skill list - what we want to match INTO
STANDARD_SKILLS = [
    "Team Leadership",
    "Python Programming",
    "Data Analysis",
    "Project Management",
    "Machine Learning",
    "Communication Skills",
    "Sales Strategy",
    "Customer Service",
]

# Pre-compute vectors for standard skills once
standard_skill_docs = {skill: nlp(skill) for skill in STANDARD_SKILLS}


def fuzzy_match_skill(phrase, threshold=0.6):
    """
    Given an arbitrary phrase (e.g. from a resume), find the closest
    standard skill using cosine similarity via spaCy vectors.
    Returns (best_match, score) or (None, best_score) if below threshold.
    """
    phrase_doc = nlp(phrase)

    best_match = None
    best_score = 0.0

    for skill_name, skill_doc in standard_skill_docs.items():
        score = phrase_doc.similarity(skill_doc)
        if score > best_score:
            best_score = score
            best_match = skill_name

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def demo_fuzzy_matching():
    print("=== Fuzzy Skill Matcher Demo ===\n")

    test_phrases = [
        "managing a team",
        "coding in python",
        "analyzing large datasets",
        "leading cross-functional projects",
        "talking to customers about issues",
        "gardening",  # should NOT match anything well
    ]

    for phrase in test_phrases:
        match, score = fuzzy_match_skill(phrase)
        if match:
            print(f"  '{phrase}' -> matched to '{match}' (score: {score:.3f})")
        else:
            print(f"  '{phrase}' -> no confident match (best score: {score:.3f})")


if __name__ == "__main__":
    compare_documents()
    demo_fuzzy_matching()