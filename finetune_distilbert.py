"""
Week 6 - Fine-Tuning distilbert-base-uncased on Job Category Data
Takes the SAME filtered dataset from Week 3 (Kaggle resumes mapped to
Data Science / Marketing / Engineering) and fine-tunes a real transformer
model on it, then compares accuracy against the Week 3 TF-IDF baseline.

Requires: pip install datasets accelerate

Run in VS Code terminal:
    python finetune_distilbert.py

Note: This will be slow on CPU (could take 20-40+ minutes depending on
your machine). If you have a GPU, it'll be much faster automatically.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---- CONFIG ----
INPUT_CSV = "data/Resume.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "outputs/distilbert_job_classifier"

# Same category mapping as Week 3 - keep this identical for a fair comparison
CATEGORY_MAP = {
    "INFORMATION-TECHNOLOGY": "Data Science",
    "DIGITAL-MEDIA": "Marketing",
    "BUSINESS-DEVELOPMENT": "Marketing",
    "ENGINEERING": "Engineering",
}
LABEL_NAMES = ["Data Science", "Marketing", "Engineering"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}


def load_and_prepare_data():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["Category"].isin(CATEGORY_MAP.keys())].copy()
    df["target_label"] = df["Category"].map(CATEGORY_MAP)
    df["label_id"] = df["target_label"].map(LABEL_TO_ID)

    print(f"Total examples: {len(df)}")
    print(df["target_label"].value_counts())

    # Same 80/20 split, same random_state as Week 3 - important for fair comparison
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target_label"]
    )
    return train_df, test_df


def tokenize_function(examples, tokenizer):
    # Truncate long resumes to 512 tokens (distilbert's max) - this is standard practice
    return tokenizer(examples["Resume_str"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def main():
    print("Loading and preparing data...")
    train_df, test_df = load_and_prepare_data()

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df[["Resume_str", "label_id"]].rename(columns={"label_id": "label"}))
    test_dataset = Dataset.from_pandas(test_df[["Resume_str", "label_id"]].rename(columns={"label_id": "label"}))

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,     # small batch size - safer for CPU/limited memory
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning (this will take a while)...")
    trainer.train()

    print("\n=== FINAL EVALUATION ===")
    eval_results = trainer.evaluate()
    print(eval_results)

    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    # Save the fine-tuned model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nFine-tuned model saved to {OUTPUT_DIR}")

    print("\n=== COMPARISON ===")
    print("Week 3 TF-IDF + SGDClassifier accuracy: 0.91 (91%)")
    print(f"Week 6 fine-tuned distilbert accuracy: {eval_results['eval_accuracy']:.2f}")


if __name__ == "__main__":
    main()