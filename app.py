import streamlit as st
import pdfplumber
import nltk

# Ensure NLTK wordnet resources are available
nltk.download('wordnet')
nltk.download('omw-1.4')

from src.info_parser import InfoParser
from src.embedding import Embedder
from src.final_rank import Ranker
from src.db_handler import DBHandler

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Resume Ranking", layout="wide", page_icon="📝")
st.title("📝 Resume Ranking Portal")

st.sidebar.header("Instructions")
st.sidebar.info("1. Upload a Job Description (JD).\n2. Upload multiple resumes.\n3. Click **Run Ranking** to see results.")

# ---------------- Initialize Components ----------------
db, parser, embedder, ranker = DBHandler(), InfoParser(), Embedder(), Ranker()

# ---------------- Upload Section ----------------
st.markdown("### 📂 Upload Files")
col1, col2 = st.columns(2)

with col1:
    job_file = st.file_uploader("Upload Job Description (.txt/.pdf)", type=["txt", "pdf"])
with col2:
    resume_files = st.file_uploader("Upload Resumes (.txt/.pdf)", type=["txt", "pdf"], accept_multiple_files=True)

run_button = st.button("🚀 Run Resume Ranking")

# ---------------- Processing ----------------
if run_button:
    if not job_file or not resume_files:
        st.warning("⚠ Please upload both a JD and at least one Resume.")
    else:
        st.info("Processing... ⏳")

        # --- JD Handling ---
        if job_file.type == "application/pdf":
            with pdfplumber.open(job_file) as pdf:
                job_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            job_text = job_file.read().decode("utf-8")

        job_info = parser.parse_job(job_text, job_file.name)
        session_id = db.insert_session(job_info["job_id"])
        embedder.process_job(job_info["job_id"], job_text, {**job_info, "text": job_text})

        # --- Resume Handling ---
        resumes_data = []
        for file in resume_files:
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            else:
                text = file.read().decode("utf-8")

            res_info = parser.parse_resume(text, file.name)
            embedder.process_resume(res_info["resume_id"], text, {**res_info, "text": text})
            resumes_data.append({**res_info, "text": text})

        # --- Ranking ---
        db.clear_session_rankings(session_id)
        ranker.rank_session(session_id, resumes_data=resumes_data)

        # --- Display Results ---
        st.markdown("## 🏆 Ranked Resumes")

        results = db.get_ranked_resumes(session_id, top_k=20)

        if not results:
            st.warning("⚠ No ranking results found. Please check your uploads.")
        else:
            for i, r in enumerate(results, 1):
                st.markdown(f"### {i}. {r['filename']}")
                col1, col2, col3, col4 = st.columns([1, 1, 2, 2])

                with col1:
                    st.metric("Final Score", f"{r['score']:.2f}")
                with col2:
                    st.write(f"💼 Experience: {r['experience']} yrs")
                with col3:
                    st.write(f"📧 {r['email']}")
                with col4:
                    st.write(f"📞 {r['phone']}")

        st.success("🎉 Resume ranking completed!")
        st.balloons()
