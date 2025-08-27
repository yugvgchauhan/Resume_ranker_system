from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    Handles embeddings for resumes and job descriptions.
    Uses SentenceTransformer for embeddings and stores them in memory (dicts).
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # In-memory stores
        self.resumes = {}
        self.jobs = {}

        # Load model
        print(f"[✔] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("[✔] Model loaded with embedding dimension:",
              self.model.get_sentence_embedding_dimension())

    def _embed_text(self, text: str):
        """Generate embedding vector for a given text"""
        if not text.strip():
            return None
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def process_resume(self, resume_id, text, metadata):
        """Embed a single resume and keep in memory."""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"[⚠] Skipping empty resume {metadata.get('filename')}")
            return
        self.resumes[resume_id] = {"embedding": embedding, "text": text, "meta": metadata}
        print(f"[✔] Resume embedded → {metadata.get('filename')} (resume_id={resume_id})")

    def process_job(self, job_id, text, metadata):
        """Embed a single job description and keep in memory."""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"[⚠] Skipping empty job {metadata.get('job_name')}")
            return
        self.jobs[job_id] = {"embedding": embedding, "text": text, "meta": metadata}
        print(f"[✔] Job embedded → {metadata.get('job_name')} (job_id={job_id})")

    def process_session(self, session_id, resumes_data, job_data):
        """Embed resumes + job for the given session."""
        print(f"\n[Embedding] Processing session {session_id}...")
        self.process_job(job_data["job_id"], job_data.get("text", ""), job_data)
        for res in resumes_data:
            self.process_resume(res["resume_id"], res.get("text", ""), res)
        print(f"[✔] Finished embeddings for session {session_id}")
