import os
import pdfplumber
from src.info_parser import InfoParser
from src.embedding import Embedder
from src.final_rank import Ranker
from src.db_handler import DBHandler


def load_text_from_file(file_path):
    """
    Load text from a TXT or PDF file.
    Used only in console pipeline testing (main.py).
    """
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def run_pipeline(jd_folder="DATA/jobs_cleaned", resume_folder="DATA/resumes_cleaned"):
    print("🚀 Starting Resume Ranking Pipeline...")

    # ---------------- Initialize Components ----------------
    db = DBHandler()
    parser = InfoParser()
    embedder = Embedder()
    ranker = Ranker()

    # ---------------- Step 1: Load JD ----------------
    if not os.path.exists(jd_folder) or not os.listdir(jd_folder):
        print(f"[⚠] No job descriptions found in {jd_folder}")
        return

    jd_file = os.listdir(jd_folder)[0]  # take first JD for pipeline test
    jd_path = os.path.join(jd_folder, jd_file)
    jd_text = load_text_from_file(jd_path)

    print(f"[JD] Loaded: {jd_file}")
    job_info = parser.parse_job(jd_text, jd_file)

    # ---------------- Step 2: Load Resumes ----------------
    if not os.path.exists(resume_folder) or not os.listdir(resume_folder):
        print(f"[⚠] No resumes found in {resume_folder}")
        return

    resumes_data = []
    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)
        text = load_text_from_file(file_path)
        res_info = parser.parse_resume(text, file)
        resumes_data.append({**res_info, "text": text})

    print(f"[Resumes] Loaded {len(resumes_data)} resumes")

    # ---------------- Step 3: Embed JD + Resumes ----------------
    embedder.process_job(job_info["job_id"], jd_text, {**job_info, "text": jd_text})
    for res in resumes_data:
        embedder.process_resume(res["resume_id"], res["text"], res)

    # ---------------- Step 4: Rank Resumes ----------------
    session_id = job_info["session_id"]
    db.clear_session_rankings(session_id)  # clear old scores for this JD
    ranker.rank_session(session_id, resumes_data=resumes_data)

    # ---------------- Step 5: Display Results ----------------
    results = db.get_ranked_resumes(session_id)

    print("\n🏆 Final Ranking Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['filename']} | Final Score={r['score']:.3f} | "
              f"Exp={r['experience']} yrs | Email={r['email']} | Phone={r['phone']}")


if __name__ == "__main__":
    run_pipeline()
