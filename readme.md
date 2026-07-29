# ResuMind — Intelligent Resume Processing Engine

An end-to-end NLP system that parses resumes, extracts structured information, and scores candidate-job fit using a combination of classical ML and transformer-based deep learning — built over an 8-week self-directed internship roadmap.

## Live Demo
[Try ResuMind live](https://resumind-f4o5lpwbkp5grku8ndmuzz.streamlit.app/)

## What It Does
Upload a resume (PDF) and paste a job description, and ResuMind will:
- Extract candidate name, email, and phone number
- Identify hard skills using rule-based pattern matching
- Compute an overall job match score combining skill overlap and semantic similarity
- Generate a skill gap analysis showing which required skills are present vs. missing

## Tech Stack
`spaCy` · `scikit-learn` · `transformers` (Hugging Face) · `sentence-transformers` · `Streamlit` · `PyMuPDF`

## Project Architecture

This project combines two complementary approaches, as outlined in the original roadmap:
- **The Fast Brain (spaCy)** — rule-based, high-precision extraction for structured, predictable data (emails, names, known skill keywords)
- **The Deep Brain (Transformers)** — contextual understanding for tasks that need semantic reasoning (job category classification, skill/title similarity, semantic search)

## Development Journey (8 Weeks)

| Week | Deliverable | Key Result |
|---|---|---|
| 1 | Resume token analyzer — tokens, lemmas, POS tags, NER, CSV export | Working pipeline on 20 sample resumes |
| 2 | Rule-based skill & job title extractor (spaCy Matcher/PhraseMatcher) | High-precision keyword-based extraction |
| 3 | Job category classifier — TF-IDF + SGDClassifier | **91% accuracy** on Kaggle Resume Dataset (Data Science/Marketing/Engineering) |
| 4 | Word vectors & fuzzy skill matching (`doc.similarity()`) | Demonstrated both the utility and limitations of averaged word vectors |
| 5 | Hugging Face `pipeline()` API — sentiment, NER, summarization, zero-shot classification | Zero-shot classification correctly categorized job descriptions with no training data |
| 6 | Fine-tuned `distilbert-base-uncased` on the same Week 3 dataset | **93.4% accuracy** — outperformed the TF-IDF baseline, especially on the previously weak Data Science category (recall improved from 0.75 → 0.92) |
| 7 | Semantic search (sentence-transformers) + custom NER fine-tuning via weak supervision | Semantic search surfaced contextually relevant resume sentences; custom NER reached 0.68 F1 for SKILL entities after addressing severe class imbalance with weighted loss |
| 8 | **ResuMind capstone** — full integration + Streamlit UI | Working end-to-end app: upload → extract → match → gap analysis |

## Notable Engineering Challenges Solved
- **Dataset sourcing**: Practice dataset (20 Newsgroups) had environment issues; pivoted to the Kaggle Resume Dataset (2,400+ real resumes) and built a category-mapping layer to align its labels with project-specific target categories.
- **Class imbalance in custom NER (Week 7)**: Initial fine-tuning collapsed to predicting only the majority "O" class (0% entity F1) due to a 62:1 token imbalance. Fixed with a custom weighted loss function, tuned iteratively (raw inverse-frequency weights overcorrected into high false-positive rates; square-root dampening with capping produced a balanced 0.68 F1).
- **Weak supervision for NER**: No manually annotated token-level data was available. Solved by using the Week 2 rule-based matchers to auto-generate training labels, allowing the fine-tuned model to learn to generalize beyond the original rule list.

## Setup & Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd resume_nlp_project

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

# Run the app
streamlit run app.py
```

## Project Structure
```
resume_nlp_project/
├── data/
│   ├── resumes_raw/              # personal sample resumes (not tracked in git)
│   └── Resume.csv                # Kaggle dataset (not tracked in git)
├── outputs/                      # trained models, CSVs (not tracked in git)
├── resume_token_analyzer.py      # Week 1
├── skill_extractor.py            # Week 2
├── train_classifier.py           # Week 3
├── similarity_matcher.py         # Week 4
├── transformers_intro.py         # Week 5
├── finetune_distilbert.py        # Week 6
├── semantic_search.py            # Week 7 - Part A
├── custom_ner_finetune.py        # Week 7 - Part B
├── app.py                        # Week 8 - Capstone Streamlit app
└── requirements.txt
```

## Known Limitations
- Phone number extraction depends on standard formatting; not all formats are currently covered.
- Custom NER model (Week 7) is trained on weakly-supervised (auto-generated, not hand-labeled) data — reasonable but not production-grade precision.
- Title standardization and full explainable-score weighting (per the original spec) are implemented as functions but not yet fully wired into the live UI — a natural next step.

## Future Improvements
- Wire in the fine-tuned NER model (Week 7) as an additional skill-extraction signal alongside the rule-based matcher
- Add zero-shot soft-skill inference (Week 5 concept) to catch implied skills not in the hard-coded skill list
- Expand the standard job title list and connect title standardization to the live UI
- Address the phone number extraction edge case
