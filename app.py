import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

st.set_page_config(page_title="Resume Matcher")

st.title("📄 Resume – Job Description Matcher")

resume_text = st.text_area("Paste Resume Text")
jd_text = st.text_area("Paste Job Description")

if st.button("Check Match"):
    if resume_text and jd_text:
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        match_percentage = round(similarity * 100, 2)

        st.success(f"Match Percentage: {match_percentage}%")

        feature_names = vectorizer.get_feature_names_out()
        resume_vec = vectors[0].toarray()[0]
        jd_vec = vectors[1].toarray()[0]

        matched = [feature_names[i] for i in range(len(feature_names))
                   if resume_vec[i] > 0 and jd_vec[i] > 0]

        missing = [feature_names[i] for i in range(len(feature_names))
                   if resume_vec[i] == 0 and jd_vec[i] > 0]

        st.subheader("✅ Matched Keywords")
        st.write(matched[:20])

        st.subheader("❌ Missing Keywords")
        st.write(missing[:20])
    else:
        st.warning("Please enter both Resume and Job Description.")
