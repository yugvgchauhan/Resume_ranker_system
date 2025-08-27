import os
import chromadb
from sentence_transformers import SentenceTransformer
from src.db_handler import DBHandler


class Embedder:
    """
    Handles embeddings for resumes and job descriptions.
    Uses SentenceTransformer for embeddings and ChromaDB as vector store.
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.db = DBHandler()

        # Ensure ChromaDB directory exists
        os.makedirs("databases/chroma_db", exist_ok=True)

        # Persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(path="databases/chroma_db")

        # Separate collections for resumes and jobs
        self.resume_collection = self.chroma_client.get_or_create_collection("resumes")
        self.job_collection = self.chroma_client.get_or_create_collection("jobs")

        # ✅ Load pretrained embedding model
        print(f"[✔] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("[✔] Model loaded with embedding dimension:",
              self.model.get_sentence_embedding_dimension())

    # ------------------------- Utility -------------------------
    def _embed_text(self, text: str):
        """Generate embedding vector for a given text"""
        if not text.strip():
            return None
        return self.model.encode(text, convert_to_numpy=True).tolist()

    # ------------------------- Resume Embedding -------------------------
    def process_resume(self, resume_id, text, metadata):
        """Embed a single resume (linked to DB resume_id)."""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"[⚠] Skipping empty resume {metadata.get('filename')}")
            return

        self.resume_collection.add(
            ids=[str(resume_id)],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        print(f"[✔] Resume embedded → {metadata.get('filename')} (resume_id={resume_id})")

    # ------------------------- Job Embedding -------------------------
    def process_job(self, job_id, text, metadata):
        """Embed a single job description (linked to DB job_id)."""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"[⚠] Skipping empty job {metadata.get('job_name')}")
            return

        self.job_collection.add(
            ids=[str(job_id)],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        print(f"[✔] Job embedded → {metadata.get('job_name')} (job_id={job_id})")

    # ------------------------- Session Embedding -------------------------
    def process_session(self, session_id, resumes_data, job_data):
        """
        Embeds only the resumes + job for the given session.
        - resumes_data: list of dicts returned from InfoParser.parse_resume()
        - job_data: dict returned from InfoParser.parse_job()
        """
        print(f"\n[Embedding] Processing session {session_id}...")

        # Embed job (must include text in job_data)
        job_text = job_data.get("text", "")
        self.process_job(job_data["job_id"], job_text, job_data)

        # Embed resumes
        for res in resumes_data:
            resume_text = res.get("text", "")
            self.process_resume(res["resume_id"], resume_text, res)

        print(f"[✔] Finished embeddings for session {session_id}")


# ------------------------- Quick Test -------------------------
if __name__ == "__main__":
    embedder = Embedder()

    print("\nCollections available in ChromaDB:")
    print(embedder.chroma_client.list_collections())

    # ----------------- Embed all files in DATA folders -----------------
    resumes_cleaned = "DATA/resumes_cleaned"
    jobs_cleaned = "DATA/jobs_cleaned"

    # Process resumes
    if os.path.exists(resumes_cleaned):
        for i, filename in enumerate(os.listdir(resumes_cleaned), start=1):
            filepath = os.path.join(resumes_cleaned, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            embedder.process_resume(i, text, {"filename": filename})
    else:
        print(f"[INFO] No resumes found in {resumes_cleaned}")

    # Process jobs
    if os.path.exists(jobs_cleaned):
        for j, filename in enumerate(os.listdir(jobs_cleaned), start=1):
            filepath = os.path.join(jobs_cleaned, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            embedder.process_job(j, text, {"job_name": filename})
    else:
        print(f"[INFO] No jobs found in {jobs_cleaned}")

    print("\nResumes in collection:", embedder.resume_collection.count())
    print("Jobs in collection:", embedder.job_collection.count())

    # Fetch one job and one resume with embeddings
    print("\nChecking one stored job embedding:")
    job_data = embedder.job_collection.get(include=["embeddings"], limit=1)
    if job_data and "embeddings" in job_data and len(job_data["embeddings"]) > 0:
        print("Embedding length (job):", len(job_data["embeddings"][0]))

    print("\nChecking one stored resume embedding:")
    resume_data = embedder.resume_collection.get(include=["embeddings"], limit=1)
    if resume_data and "embeddings" in resume_data and len(resume_data["embeddings"]) > 0:
        print("Embedding length (resume):", len(resume_data["embeddings"][0]))
