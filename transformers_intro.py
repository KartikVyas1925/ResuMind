"""
Week 5 - Introduction to Transformers & Hugging Face
Uses the pipeline() API to run inference with pre-trained transformer
models - no training required. Covers sentiment analysis, NER,
summarization, and zero-shot classification (the key one for your project).

Run in VS Code terminal:
    python transformers_intro.py

Note: First run will download model weights (a few hundred MB each) -
this only happens once, then they're cached locally.
"""

from transformers import pipeline


def demo_sentiment_analysis():
    print("=== Sentiment Analysis ===\n")
    classifier = pipeline("sentiment-analysis")

    texts = [
        "This candidate has excellent leadership skills and a strong track record.",
        "The resume was poorly formatted and lacked relevant experience.",
    ]
    for text in texts:
        result = classifier(text)[0]
        print(f"  '{text}'")
        print(f"    -> {result['label']} (confidence: {result['score']:.3f})\n")


def demo_ner():
    print("=== Named Entity Recognition (Transformer-based) ===\n")
    ner = pipeline("ner", aggregation_strategy="simple")

    text = "John Smith worked at Google in California before joining Microsoft in 2020."
    results = ner(text)
    for ent in results:
        print(f"  {ent['word']} -> {ent['entity_group']} (confidence: {ent['score']:.3f})")
    print("\nCompare this to spaCy's NER from Week 1 - similar concept, different model.\n")


def demo_summarization():
    print("=== Summarization ===\n")
    summarizer = pipeline("summarization")

    long_text = """
    Our company is looking for a Senior Data Scientist to join our growing analytics team.
    The ideal candidate will have 5+ years of experience in machine learning, statistical
    modeling, and data visualization. You will work closely with product and engineering
    teams to build predictive models that drive business decisions. Strong programming
    skills in Python and SQL are required. Experience with cloud platforms like AWS or
    Azure is a plus. This role offers the opportunity to work on cutting-edge problems
    in a fast-paced, collaborative environment.
    """

    summary = summarizer(long_text, max_length=50, min_length=20, do_sample=False)
    print(f"  Original length: {len(long_text.split())} words")
    print(f"  Summary: {summary[0]['summary_text']}\n")


def demo_zero_shot_classification():
    """
    THIS IS THE KEY ONE FOR YOUR PROJECT.
    Zero-shot means: no training data needed. You give it any text AND
    any list of candidate labels, and it scores how well each label fits -
    even labels it has never explicitly seen before.
    """
    print("=== Zero-Shot Classification (key for skill categorization) ===\n")
    classifier = pipeline("zero-shot-classification")

    # Example 1: classify a skill/phrase into high-level categories
    text1 = "led a team of 5 engineers to deliver the project on time"
    candidate_labels_1 = ["leadership", "project management", "agile methodology", "technical writing"]

    result1 = classifier(text1, candidate_labels_1)
    print(f"  Text: '{text1}'")
    for label, score in zip(result1["labels"], result1["scores"]):
        print(f"    {label}: {score:.3f}")

    # Example 2: classify a job description into your 3 target categories
    print()
    text2 = "We are looking for someone to build predictive models and analyze large datasets using Python."
    candidate_labels_2 = ["Data Science", "Marketing", "Engineering"]

    result2 = classifier(text2, candidate_labels_2)
    print(f"  Text: '{text2}'")
    for label, score in zip(result2["labels"], result2["scores"]):
        print(f"    {label}: {score:.3f}")
    print(f"\n  Best match: {result2['labels'][0]} (no training data was used for this!)")


if __name__ == "__main__":
    demo_sentiment_analysis()
    demo_ner()
    demo_summarization()
    demo_zero_shot_classification()