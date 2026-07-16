"""
Week 7 Part B - Custom NER Fine-Tuning (SKILL, JOB_TITLE)
Uses WEAK SUPERVISION: Week 2's rule-based matchers auto-generate token
labels, then we fine-tune distilbert for token classification on those
auto-generated labels. This avoids manual annotation of hundreds of resumes.

Requires: pip install seqeval

Run in VS Code terminal:
    python custom_ner_finetune.py

Note: Like Week 6, this is a real fine-tuning run - expect 20-40+ minutes
on CPU.
"""

import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import accuracy_score as seqeval_accuracy
import torch
import torch.nn as nn

# ---- CONFIG ----
INPUT_CSV = "data/Resume.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "outputs/distilbert_ner"
MAX_EXAMPLES = 300  # keep training time reasonable - increase later if you want

nlp = spacy.load("en_core_web_sm")

# Reuse the same skill list and title patterns from Week 2
SKILL_LIST = [
    "Python", "SQL", "Java", "JavaScript", "C++", "Machine Learning",
    "Deep Learning", "Project Management", "Data Analysis", "Excel",
    "Tableau", "Power BI", "AWS", "Azure", "Docker", "Kubernetes",
    "Git", "Agile", "Scrum", "Leadership", "Communication",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "React", "Node.js", "HTML", "CSS", "Marketing", "SEO",
]
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILL", [nlp.make_doc(s) for s in SKILL_LIST])

title_matcher = Matcher(nlp.vocab)
title_matcher.add("JOB_TITLE", [
    [{"LOWER": "chief"}, {"LOWER": "executive"}, {"LOWER": "officer"}],
    [{"LOWER": {"IN": ["senior", "junior", "lead"]}, "OP": "?"},
     {"POS": "NOUN", "OP": "*"}, {"LOWER": "manager"}],
    [{"LOWER": {"IN": ["senior", "junior", "lead"]}, "OP": "?"},
     {"POS": "NOUN", "OP": "*"}, {"LOWER": "engineer"}],
])

LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-TITLE", "I-TITLE"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def auto_label_text(text):
    """
    Runs Week 2's matchers on text and produces BIO-tagged tokens.
    Returns (tokens, tags) - the weak-supervision training example.
    """
    doc = nlp(text)
    tags = ["O"] * len(doc)

    skill_matches = skill_matcher(doc)
    title_matches = title_matcher(doc)

    for match_id, start, end in skill_matches:
        tags[start] = "B-SKILL"
        for i in range(start + 1, end):
            tags[i] = "I-SKILL"

    for match_id, start, end in title_matches:
        # don't overwrite existing skill tags
        if tags[start] == "O":
            tags[start] = "B-TITLE"
        for i in range(start + 1, end):
            if tags[i] == "O":
                tags[i] = "I-TITLE"

    tokens = [t.text for t in doc]
    return tokens, tags


def build_weak_labeled_dataset():
    print("Loading resumes and auto-labeling with Week 2 matchers...")
    df = pd.read_csv(INPUT_CSV)
    df = df.sample(n=min(MAX_EXAMPLES, len(df)), random_state=42)  # cap for reasonable training time

    all_tokens, all_tags = [], []
    for text in df["Resume_str"].astype(str):
        # Only use first ~200 words per resume - keeps sequences manageable
        short_text = " ".join(text.split()[:200])
        tokens, tags = auto_label_text(short_text)
        if any(t != "O" for t in tags):  # only keep examples with at least one match
            all_tokens.append(tokens)
            all_tags.append(tags)

    print(f"Built {len(all_tokens)} labeled examples (out of {len(df)} resumes)")
    label_counts = {}
    for tags in all_tags:
        for t in tags:
            label_counts[t] = label_counts.get(t, 0) + 1
    print("Label distribution:", label_counts)

    return all_tokens, all_tags


def tokenize_and_align_labels(examples, tokenizer):
    """
    Standard Hugging Face pattern: BERT splits words into subwords, so we
    need to align our word-level labels to the subword tokens. Only the
    FIRST subword of each word gets the real label; the rest get -100
    (ignored during loss calculation).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=256
    )

    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL_TO_ID[tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to the loss function.
    Without this, the model just predicts 'O' for everything since O
    tokens vastly outnumber entity tokens (~62:1 in this dataset).
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, len(LABEL_LIST)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(all_tags):
    """Computes inverse-frequency weights so rare entity classes matter more."""
    label_counts = {label: 0 for label in LABEL_LIST}
    for tags in all_tags:
        for t in tags:
            label_counts[t] += 1

    total = sum(label_counts.values())
    weights = []
    for label in LABEL_LIST:
        count = max(label_counts[label], 1)  # avoid divide by zero
        raw_weight = total / (len(LABEL_LIST) * count)
        dampened_weight = raw_weight ** 0.5   # sqrt dampening - softens extreme swings
        capped_weight = min(dampened_weight, 10.0)  # cap so no class dominates the loss
        weights.append(capped_weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"\nClass weights: {dict(zip(LABEL_LIST, [round(w, 2) for w in weights]))}")
    return weights_tensor


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID_TO_LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID_TO_LABEL[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {"accuracy": seqeval_accuracy(true_labels, true_predictions)}


def main():
    tokens, tags = build_weak_labeled_dataset()

    train_tokens, test_tokens, train_tags, test_tags = train_test_split(
        tokens, tags, test_size=0.2, random_state=42
    )

    train_dataset = Dataset.from_dict({"tokens": train_tokens, "tags": train_tags})
    test_dataset = Dataset.from_dict({"tokens": test_tokens, "tags": test_tags})

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    )

    print("Tokenizing and aligning labels...")
    train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    class_weights = compute_class_weights(train_tags)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\n=== FINAL EVALUATION ===")
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID_TO_LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID_TO_LABEL[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    print(seqeval_report(true_labels, true_predictions))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()