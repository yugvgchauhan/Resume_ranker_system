import os
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

    # ---------------- Semantic Similarity (Stored Embeddings) ----------------
    def compute_semantic_similarity(self, job_id, resume_id):
        """Compute cosine similarity using embeddings already in ChromaDB"""
        job_data = self.embedder.job_collection.get(ids=[str(job_id)], include=["embeddings"])
        resume_data = self.embedder.resume_collection.get(ids=[str(resume_id)], include=["embeddings"])

        # Check if embeddings exist and are non-empty
        if job_data is None or "embeddings" not in job_data or len(job_data["embeddings"]) == 0:
            return 0.0
        if resume_data is None or "embeddings" not in resume_data or len(resume_data["embeddings"]) == 0:
            return 0.0

        job_vec = np.array(job_data["embeddings"][0]).reshape(1, -1)
        resume_vec = np.array(resume_data["embeddings"][0]).reshape(1, -1)

        return float(cosine_similarity(job_vec, resume_vec)[0][0])

    # ---------------- Keyword Overlap (with Synonyms) ----------------
    def compute_keyword_overlap(self, job_text, resume_text):
        """Expand job text with synonyms, then compute keyword overlap %"""
        job_expanded = self.preprocessor.expand_with_synonyms(job_text.lower())
        resume_text = resume_text.lower()

        job_keywords = set(job_expanded.split())
        resume_words = set(resume_text.split())

        if not job_keywords:
            return 0.0
        overlap = len(job_keywords & resume_words) / len(job_keywords)
        return overlap

    # ---------------- Final Scoring ----------------
    def compute_score(self, job_id, resume_id, job_text, resume_text, alpha=0.7):
        """
        Combine semantic similarity + keyword overlap
        alpha = weight for semantic (default 0.7)
        (1-alpha) = weight for keyword overlap
        """
        semantic = self.compute_semantic_similarity(job_id, resume_id)
        keyword = self.compute_keyword_overlap(job_text, resume_text)
        return alpha * semantic + (1 - alpha) * keyword

    # ---------------- Rank Resumes ----------------
    def rank_session(self, session_id, resumes_data):
        """
        Rank all resumes in the session against the job description
        - resumes_data: list of dicts {resume_id, text, ...}
        """
        # Get job metadata for session
        job_row = self.db.get_job_by_session(session_id)
        if not job_row:
            print(f"[⚠] No job found for session {session_id}")
            return

        job_id = job_row["job_id"]
        job_name = job_row["job_name"]

        # ✅ Fetch JD text from Chroma (not DB)
        job_data = self.embedder.job_collection.get(ids=[str(job_id)], include=["documents"])
        job_text = job_data["documents"][0] if job_data["documents"] else ""

        if not job_text:
            print(f"[⚠] No job text found for job_id={job_id}")
            return

        print(f"[Ranking] Job = {job_name} (job_id={job_id})")

        ranked = []
        for res in resumes_data:
            resume_id = res["resume_id"]
            resume_text = res["text"]

            score = self.compute_score(job_id, resume_id, job_text, resume_text)
            ranked.append((resume_id, score))

        # Sort descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Store results in DB
        # ⚠️ Make sure DBHandler has a method to store rankings
        if hasattr(self.db, "store_rankings"):
            self.db.store_rankings(session_id, ranked)
        else:
            print("[⚠] DBHandler has no store_rankings(). Please implement it if needed.")

        print(f"[✔] Ranking completed for session {session_id}")


# ----------------- Quick Test: rank all jobs vs all resumes -----------------
if __name__ == "__main__":
    ranker = Ranker()

    jobs_data = ranker.embedder.job_collection.get(include=["documents", "metadatas"], limit=50)
    resumes_data = ranker.embedder.resume_collection.get(include=["documents", "metadatas"], limit=200)

    if not jobs_data["ids"] or not resumes_data["ids"]:
        print("[⚠] No jobs or resumes found in Chroma. Please run embedding first.")
    else:
        for j, job_id in enumerate(jobs_data["ids"]):
            job_text = jobs_data["documents"][j]
            job_name = jobs_data["metadatas"][j].get("job_name", f"job_{job_id}")

            print(f"\n[TEST] Ranking resumes against Job: {job_name} (Job ID={job_id})")

            results = []
            for i, res_id in enumerate(resumes_data["ids"]):
                resume_text = resumes_data["documents"][i]
                metadata = resumes_data["metadatas"][i]
                filename = metadata.get("filename", f"resume_{res_id}")

                sim = ranker.compute_semantic_similarity(job_id, res_id)
                overlap = ranker.compute_keyword_overlap(job_text, resume_text)
                score = ranker.compute_score(job_id, res_id, job_text, resume_text)

                results.append((res_id, filename, sim, overlap, score))

            # Sort by score
            results.sort(key=lambda x: x[4], reverse=True)

            # Print top 3 resumes per job
            print("🏆 Top Ranked Resumes:")
            for rank, (rid, fname, sim, overlap, score) in enumerate(results[:3], 1):
                print(f"{rank}. {fname} (Resume ID={rid}) | Semantic={sim:.3f} | Overlap={overlap:.3f} | Final={score:.3f}")
