import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.db_handler import DBHandler
from src.preprocessing import Preprocessor
from src.embedding import Embedder

class Ranker:
    def __init__(self):
        self.db = DBHandler()
        self.preprocessor = Preprocessor()
        self.embedder = Embedder()

    def compute_semantic_similarity(self, job_id, resume_id):
        """Compute cosine similarity using in-memory embeddings"""
        job = self.embedder.jobs.get(job_id)
        resume = self.embedder.resumes.get(resume_id)

        if not job or not resume:
            return 0.0

        job_vec = np.array(job["embedding"]).reshape(1, -1)
        resume_vec = np.array(resume["embedding"]).reshape(1, -1)
        return float(cosine_similarity(job_vec, resume_vec)[0][0])

    def compute_keyword_overlap(self, job_text, resume_text):
        """Expand job text with synonyms, then compute keyword overlap %"""
        job_expanded = self.preprocessor.expand_with_synonyms(job_text.lower())
        resume_text = resume_text.lower()
        job_keywords = set(job_expanded.split())
        resume_words = set(resume_text.split())
        if not job_keywords:
            return 0.0
        return len(job_keywords & resume_words) / len(job_keywords)

    def compute_score(self, job_id, resume_id, job_text, resume_text, alpha=0.7):
        """Combine semantic similarity + keyword overlap"""
        semantic = self.compute_semantic_similarity(job_id, resume_id)
        keyword = self.compute_keyword_overlap(job_text, resume_text)
        return alpha * semantic + (1 - alpha) * keyword

    def rank_session(self, session_id, resumes_data):
        """Rank all resumes in the session against the job description"""
        job_row = self.db.get_job_by_session(session_id)
        if not job_row:
            print(f"[⚠] No job found for session {session_id}")
            return

        job_id = job_row["job_id"]
        job_name = job_row["job_name"]

        job = self.embedder.jobs.get(job_id)
        if not job:
            print(f"[⚠] No job embedding found for job_id={job_id}")
            return
        job_text = job["text"]

        print(f"[Ranking] Job = {job_name} (job_id={job_id})")

        ranked = []
        for res in resumes_data:
            resume_id = res["resume_id"]
            resume_text = res["text"]
            score = self.compute_score(job_id, resume_id, job_text, resume_text)
            ranked.append((resume_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        if hasattr(self.db, "store_rankings"):
            self.db.store_rankings(session_id, ranked)
        print(f"[✔] Ranking completed for session {session_id}")
