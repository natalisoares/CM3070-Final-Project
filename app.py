import streamlit as st
import joblib
import fitz
import json
import base64
from docx import Document
from scipy.sparse import hstack, csr_matrix
import spacy
from auth.auth import get_openai_client

# OpenAI API 
client = get_openai_client()

# Page config
st.set_page_config(page_title="Research Classifier", layout="wide")

# Styles
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# NLP and models
nlp = spacy.load("en_core_web_sm")
disc_model = joblib.load("models/discipline_model.joblib")
subf_model = joblib.load("models/subfield_model.joblib")
meth_model = joblib.load("models/methodology_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

with open("data/methodology_keywords.json") as f:
    meth_keywords = list({kw.lower() for kws in json.load(f).values() for kw in kws})

with open("data/discipline_mapping.json") as f:
    discipline_mapping = json.load(f)

# Session state defaults
defaults = {
    "classified": False,
    "file": None,
    "text": None,
    "title": "",
    "tools": "",
    "top_kw": [],
    "labels": {},
    "explanation": None,
    "extracted_terms": ""
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# File type detect
def read_text(file):
    if file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.getvalue(), filetype="pdf")
        return "\n".join(p.get_text() for p in pdf)
    elif file.name.endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return ""

# Preview for pdf
def render_pdf(file, max_pages=20):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for i, page in enumerate(doc[:max_pages]):
        pix = page.get_pixmap(dpi=100)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        st.image(f"data:image/png;base64,{b64}", use_column_width=True)

# Title for classification output
def get_title(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = " ".join(lines[:2])
    return title if title else "This paper"

# Noun filtering
def nlp_data(text):
    doc = nlp(text)
    tokens = [t.text for t in doc if t.pos_ in {"PROPN", "NOUN"}]
    hits = [w for w in tokens if any(kw in w.lower() for kw in ["dataset", "tool", "library", "framework", "metric", "platform", "environment"])]
    return ", ".join(sorted(set(hits))[:15])

# Classification
def classify(text):
    # Convert text into TF-IDF features
    tfidf_vec = tfidf.transform([text])
    
    # Create binary indicators for methodology keywords
    rule_features = [1 if kw in text.lower() else 0 for kw in meth_keywords]
    
    # Combine both feature sets
    combined_vec = hstack([tfidf_vec, csr_matrix([rule_features])])

    # Predict discipline and subfield
    discipline = disc_model.predict(tfidf_vec)[0]
    subfield_probs = subf_model.predict_proba(tfidf_vec)[0]
    subfield_labels = subf_model.classes_
    
    # Filter subfields by discipline mapping
    candidates = discipline_mapping[discipline]
    filtered_subfields = {
        label: prob for label, prob in zip(subfield_labels, subfield_probs)
        if label in candidates
    }
    subfield = max(filtered_subfields, key=filtered_subfields.get)
    
    # Predict methodology
    methodology = meth_model.predict(combined_vec)[0]

    # Extract top terms
    feature_names = tfidf.get_feature_names_out()
    top_kw = [w for w, _ in sorted(
        zip(feature_names, tfidf_vec.toarray()[0]),
        key=lambda x: -x[1]
    )[:10]]

    return discipline, subfield, methodology, top_kw

# Explanation
def explanation(text, discipline, subfield, methodology):
    # Prompt with labels and paper text
    prompt = f"""
You are evaluating a research paper and explaining its classification.

Assigned classification:
- Discipline: {discipline}
- Subfield: {subfield}
- Methodology: {methodology}

Your task is to explain why these labels are correct, using the actual text of the paper.

Instructions:
- Use real phrases and technical content from the paper
- Focus on research goals, models used, and methods
- DO NOT define general terms like 'Computer Science' or 'Quantitative'
- Write 2–4 short paragraphs of plain text

Paper:
{text[:4000]}
"""
    # Send prompt to model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You justify research classification using paper content only."},
            {"role": "user", "content": prompt}
        ]
    )
    # Return explanation
    return response.choices[0].message.content.strip().replace("\n", "<br>")

# Data and tools 
def data_tools(text, extracted_terms=""):
    # Prompt with extracted keywords and paper text
    prompt = (
        f"NLP-extracted keywords: {extracted_terms}\n\n"
        "Based on this and the paper below, extract:\n"
        "- Datasets\n- Libraries or models\n- Metrics\n- Environments\n\n"
        "Return each category as a list:\n"
        "Datasets:\n• Item 1\n• Item 2\n\nLibraries:\n• Item 1...\n\n"
        "End with a note. No markdown. No indentation.\n\n"
        "Paper:\n" + text[:4000]
    )
    # Send prompt to model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a precise technical summariser."},
            {"role": "user", "content": prompt}
        ]
    )
    # Return result
    return response.choices[0].message.content.strip().replace("\n\n", "<br>").replace("\n", "<br>")

# AI assistant
def ai_assistant(text, user_input, top_kw, extracted_terms):
    # Build prompt with top key words and extracted terms
    prompt = f"""
You are helping analyse a research paper.

Paper context:
- Top keywords: {', '.join(top_kw)}
- Mentioned tools/components: {extracted_terms}

User's question:
{user_input}

Use academic tone. Be precise and support the answer with paper context.
"""
    # Send prompt to model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for analysing academic research papers."},
            {"role": "user", "content": prompt}
        ]
    )
    # Return answer
    return response.choices[0].message.content.strip()

# UI layout
st.markdown("<h1 class='main-title'>Research Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-sub'>Research paper classification tool for students and researchers</p>", unsafe_allow_html=True)

if st.session_state.classified:
    col_home, _ = st.columns([1, 5])
    with col_home:
        if st.button("🏠 Home"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# Upload and classify
if not st.session_state.classified:
    file = st.file_uploader("Upload your research paper", type=["pdf", "docx", "txt"])
    if file:
        st.session_state.file = file

    if st.button("Start Classification"):
        text = read_text(st.session_state.file)
        st.session_state.text = text
        st.session_state.title = get_title(text)

        # Extract terms 
        extracted_terms = nlp_data(text)

        # Classify
        discipline, subfield, methodology, top_kw = classify(text)
        st.session_state.labels = {
            "discipline": discipline,
            "subfield": subfield,
            "methodology": methodology
        }
        st.session_state.top_kw = top_kw
        st.session_state.extracted_terms = extracted_terms
        st.session_state.tools = data_tools(text, extracted_terms=extracted_terms)
        st.session_state.explanation = None
        st.session_state.classified = True
        st.rerun()

# Output results
if st.session_state.classified:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        tab1, tab2 = st.tabs(["Research Paper Preview", "Chat with AI Assistant"])

        with tab1:
            if st.session_state.file.name.endswith(".pdf"):
                render_pdf(st.session_state.file)
            else:
                st.markdown("#### Full Text Preview")
                st.markdown(st.session_state.text)

        with tab2:
            st.markdown("### Ask a question about the paper")
            user_input = st.text_input("Your question:")
            if user_input:
                with st.spinner("AI is thinking..."):
                    answer = ai_assistant(
                        st.session_state.text,
                        user_input,
                        st.session_state.top_kw,
                        st.session_state.extracted_terms
                    )
                    st.markdown(answer)

    with col2:
        tab4, tab5, tab6 = st.tabs(["Classification", "Explanation", "Data & Tools"])

        with tab4:
            lbl = st.session_state.labels
            st.markdown(f"**Title:** {st.session_state.title}")
            st.markdown(f"**Discipline:** {lbl['discipline']}")
            st.markdown(f"**Subfield:** {lbl['subfield']}")
            st.markdown(f"**Methodology:** {lbl['methodology']}")
            st.markdown(f"**Top terms:** {', '.join(st.session_state.top_kw)}")

        with tab5:
            if not st.session_state.explanation:
                with st.spinner("Generating explanation..."):
                    lbl = st.session_state.labels
                    st.session_state.explanation = explanation(
                        st.session_state.text,
                        lbl["discipline"],
                        lbl["subfield"],
                        lbl["methodology"]
                    )
            st.markdown(st.session_state.explanation, unsafe_allow_html=True)

        with tab6:
            st.markdown(st.session_state.tools, unsafe_allow_html=True)
