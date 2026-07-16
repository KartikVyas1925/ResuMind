"""
ResuMind - Intelligent Resume Processing Engine (Capstone MVP)
Integrates:
  - Week 1: spaCy structural extraction (name, email)
  - Week 2: rule-based skill/title extraction
  - Week 4/7: sentence-transformers for semantic similarity & title standardization
  - Week 3/6: category classification (optional display)

Run in VS Code terminal:
    streamlit run app.py
"""

import re
import streamlit as st
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from sentence_transformers import SentenceTransformer, util

# ---- Load models once (cached so Streamlit doesn't reload on every interaction) ----
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sbert

nlp, sbert = load_models()

# ---- Reuse Week 2's skill list & matchers ----
SKILL_LIST = [
    "Python", "SQL", "Java", "JavaScript", "C++", "Machine Learning",
    "Deep Learning", "Project Management", "Data Analysis", "Excel",
    "Tableau", "Power BI", "AWS", "Azure", "Docker", "Kubernetes",
    "Git", "Agile", "Scrum", "Leadership", "Communication",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "React", "Node.js", "HTML", "CSS", "Marketing", "SEO",
    "Content Strategy", "Salesforce", "CRM", "Six Sigma",
]
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILL", [nlp.make_doc(s) for s in SKILL_LIST])

# Standard job titles for standardization (Week 7 concept - expand this list as needed)
STANDARD_TITLES = [
    "Senior Data Scientist", "Data Analyst", "Machine Learning Engineer",
    "Software Engineer", "Senior Software Engineer", "Marketing Manager",
    "Digital Marketing Specialist", "Project Manager", "Product Manager",
    "Mechanical Engineer", "Business Analyst", "Sales Manager",
]
standard_title_embeddings = sbert.encode(STANDARD_TITLES, convert_to_tensor=True)


# ---------------------------------------------------------
# Extraction functions (Week 1 + Week 2)
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    import fitz
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else "Not found"


def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    return match.group(0) if match else "Not found"


def extract_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Check each of the first 5 lines SEPARATELY (not joined) so the name
    # doesn't accidentally merge with the line below it
    for line in lines[:5]:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                return ent.text

    # Fallback heuristic: if spaCy found nothing, the first line is often
    # just the name on a resume (e.g. "Kartik Vyas") - use it if it looks
    # like a short name (not a full sentence or heading)
    if lines and 1 <= len(lines[0].split()) <= 4 and not lines[0].isupper():
        return lines[0]

    return "Not found"


def extract_skills(text):
    doc = nlp(text)
    matches = skill_matcher(doc)
    found = set()
    for match_id, start, end in matches:
        found.add(doc[start:end].text)
    return sorted(found)


# ---------------------------------------------------------
# Semantic functions (Week 4 + Week 7)
# ---------------------------------------------------------
def standardize_title(raw_title):
    """Finds the closest standard title using cosine similarity."""
    if not raw_title.strip():
        return "N/A", 0.0
    query_embedding = sbert.encode(raw_title, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, standard_title_embeddings)[0]
    best_idx = scores.argmax().item()
    return STANDARD_TITLES[best_idx], scores[best_idx].item()


def compute_match_score(resume_text, jd_text, resume_skills, jd_skills):
    """
    Weighted formula from the project spec:
    Total Match = (Skill Match * 0.4) + (Semantic Similarity * 0.3) + (Title Match * 0.3)
    Simplified here to Skill Match (0.5) + Semantic Similarity (0.5) for the MVP -
    title matching requires a target title input, which we skip for simplicity.
    """
    # Skill overlap score
    resume_skills_lower = set(s.lower() for s in resume_skills)
    jd_skills_lower = set(s.lower() for s in jd_skills)
    if jd_skills_lower:
        skill_overlap = len(resume_skills_lower & jd_skills_lower) / len(jd_skills_lower)
    else:
        skill_overlap = 0.0

    # Semantic similarity score (whole document level)
    resume_emb = sbert.encode(resume_text, convert_to_tensor=True)
    jd_emb = sbert.encode(jd_text, convert_to_tensor=True)
    semantic_score = util.cos_sim(resume_emb, jd_emb)[0][0].item()

    total_score = (skill_overlap * 0.5) + (semantic_score * 0.5)
    return total_score, skill_overlap, semantic_score


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="ResuMind", layout="wide")
st.title("ResuMind — Intelligent Resume Processing Engine")

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Paste Job Description", height=250)
    analyze_button = st.button("Analyze")

with col2:
    st.header("Resume Snapshot")

    if analyze_button and uploaded_resume:
        resume_text = extract_text_from_pdf(uploaded_resume)

        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        skills = extract_skills(resume_text)

        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Extracted Skills:** {', '.join(skills) if skills else 'None found'}")

        with st.expander("🔍 Debug: Raw extracted text (first 800 chars)"):
            st.text(resume_text[:800])

        st.session_state["resume_text"] = resume_text
        st.session_state["resume_skills"] = skills
    elif analyze_button and not uploaded_resume:
        st.warning("Please upload a resume PDF first.")

st.divider()
st.header("Job Match Analysis")

if analyze_button and uploaded_resume and jd_text.strip():
    resume_text = st.session_state.get("resume_text", "")
    resume_skills = st.session_state.get("resume_skills", [])
    jd_skills = extract_skills(jd_text)

    total_score, skill_overlap, semantic_score = compute_match_score(
        resume_text, jd_text, resume_skills, jd_skills
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Overall Match Score", f"{total_score * 100:.0f}%")
    col_b.metric("Skill Overlap", f"{skill_overlap * 100:.0f}%")
    col_c.metric("Semantic Similarity", f"{semantic_score * 100:.0f}%")

    st.subheader("Skill Gap Analysis")
    resume_skills_lower = set(s.lower() for s in resume_skills)
    for skill in jd_skills:
        icon = "✅" if skill.lower() in resume_skills_lower else "❌"
        st.write(f"{icon} {skill}")

elif analyze_button and not jd_text.strip():
    st.info("Paste a job description above to see match analysis.")