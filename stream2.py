import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load assessments
with open("assessments_with_embeddings.json", "r") as f:
    assessments = json.load(f)

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings on-the-fly
for assessment in assessments:
    if "embedding" not in assessment:
        assessment["embedding"] = model.encode(assessment["description"]).tolist()

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")

input_mode = st.radio("Choose Input Type", ["Text", "Job Description URL"])

if input_mode == "Text":
    user_input = st.text_area("Paste your job description or role here:")
else:
    user_url = st.text_input("Enter the URL of a job description:")
    if user_url:
        try:
            response = requests.get(user_url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            user_input = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            st.error(f"Failed to fetch content from URL: {e}")
            user_input = ""

if st.button("Recommend Assessments") and user_input:
    input_embedding = model.encode(user_input).reshape(1, -1)
    scored = []
    for assmnt in assessments:
        score = cosine_similarity(input_embedding, np.array(assmnt["embedding"]).reshape(1, -1))[0][0]
        scored.append((score, assmnt))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_matches = scored[:10]

    st.subheader("Top Recommended Assessments")
    for score, a in top_matches:
        st.markdown(f"""
        **[{a['name']}]({a['url']})**
        - 🕒 Duration: {a['duration']}
        - 🧪 Test Type: {a['test_type']}
        - 🌐 Remote Testing: {a['remote_testing']}
        - 📊 Adaptive/IRT: {a['adaptive_irt']}
        """)
