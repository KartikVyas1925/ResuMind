"""
Week 3 - Real Classifier: Job Category Classification
Filters the Kaggle Resume Dataset down to 3 target categories,
relabels them, and trains + evaluates a TF-IDF + SGDClassifier pipeline.

Run in VS Code terminal:
    python train_classifier.py
"""

import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---- CONFIG ----
INPUT_CSV = "data/Resume.csv"
OUTPUT_MODEL = "outputs/job_classifier.pkl"
OUTPUT_VECTORIZER = "outputs/tfidf_vectorizer.pkl"

# Map Kaggle's actual categories to your 3 target categories.
# Kaggle categories are ALL CAPS in the raw file - check yours and adjust if needed.
CATEGORY_MAP = {
    "INFORMATION-TECHNOLOGY": "Data Science",
    "DIGITAL-MEDIA": "Marketing",
    "BUSINESS-DEVELOPMENT": "Marketing",
    "ENGINEERING": "Engineering",
}

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def preprocess(text):
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(tokens)


def main():
    os.makedirs("outputs", exist_ok=True)

    # ---- Load and filter ----
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total resumes in dataset: {len(df)}")
    print(f"Available categories: {df['Category'].unique()}")

    # Filter to only the categories we care about
    df = df[df["Category"].isin(CATEGORY_MAP.keys())].copy()
    df["target_label"] = df["Category"].map(CATEGORY_MAP)
    print(f"\nFiltered to {len(df)} resumes across target categories")
    print(df["target_label"].value_counts())

    # ---- Preprocess ----
    print("\nPreprocessing text with spaCy (this may take a few minutes)...")
    df["clean_text"] = [
        " ".join([t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space])
        for doc in nlp.pipe(df["Resume_str"].astype(str), batch_size=50)
    ]

    # ---- Train/test split ----
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["target_label"], test_size=0.2, random_state=42, stratify=df["target_label"]
    )

    # ---- Vectorize ----
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # ---- Train ----
    print("\nTraining classifier...")
    clf = SGDClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = clf.predict(X_test)
    print("\n=== RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ---- Save model + vectorizer for later use (e.g. in ResuMind capstone) ----
    joblib.dump(clf, OUTPUT_MODEL)
    joblib.dump(vectorizer, OUTPUT_VECTORIZER)
    print(f"\nModel saved to {OUTPUT_MODEL}")
    print(f"Vectorizer saved to {OUTPUT_VECTORIZER}")


if __name__ == "__main__":
    main()