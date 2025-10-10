"""
Fallback embedding system that works without ChromaDB
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from src.db_handler import DBHandler


class EmbedderFallback:
    """
    Fallback embedder that stores embeddings in memory when ChromaDB fails
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.db = DBHandler()
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded with embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # In-memory storage for embeddings
        self.resume_embeddings = {}
        self.job_embeddings = {}
        self.resume_documents = {}
        self.job_documents = {}
        
        print("Fallback embedder initialized (in-memory storage)")

    def _embed_text(self, text: str):
        """Generate embedding vector for a given text"""
        if not text.strip():
            return None
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def process_resume(self, resume_id, text, metadata):
        """Embed a single resume and store in memory"""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"Skipping empty resume {metadata.get('filename')}")
            return

        self.resume_embeddings[str(resume_id)] = embedding
        self.resume_documents[str(resume_id)] = {
            'text': text,
            'metadata': metadata
        }
        
        print(f"Resume embedded (in-memory) -> {metadata.get('filename')} (resume_id={resume_id})")

    def process_job(self, job_id, text, metadata):
        """Embed a single job description and store in memory"""
        embedding = self._embed_text(text)
        if embedding is None:
            print(f"Skipping empty job {metadata.get('job_name')}")
            return

        self.job_embeddings[str(job_id)] = embedding
        self.job_documents[str(job_id)] = {
            'text': text,
            'metadata': metadata
        }
        
        print(f"Job embedded (in-memory) -> {metadata.get('job_name')} (job_id={job_id})")

    def get_resume_embedding(self, resume_id):
        """Get resume embedding by ID"""
        return self.resume_embeddings.get(str(resume_id))

    def get_job_embedding(self, job_id):
        """Get job embedding by ID"""
        return self.job_embeddings.get(str(job_id))

    def get_resume_document(self, resume_id):
        """Get resume document by ID"""
        return self.resume_documents.get(str(resume_id))

    def get_job_document(self, job_id):
        """Get job document by ID"""
        return self.job_documents.get(str(job_id))

    def compute_similarity(self, job_id, resume_id):
        """Compute cosine similarity between job and resume"""
        job_embedding = self.get_job_embedding(job_id)
        resume_embedding = self.get_resume_embedding(resume_id)
        
        if job_embedding is None or resume_embedding is None:
            return 0.0
        
        # Convert to numpy arrays
        job_vec = np.array(job_embedding).reshape(1, -1)
        resume_vec = np.array(resume_embedding).reshape(1, -1)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity(job_vec, resume_vec)[0][0])


# Test function
if __name__ == "__main__":
    embedder = EmbedderFallback()
    
    # Test with sample data
    sample_job = "Python developer with Django experience"
    sample_resume = "Experienced Python developer with Django and Flask skills"
    
    embedder.process_job(1, sample_job, {"job_name": "test_job"})
    embedder.process_resume(1, sample_resume, {"filename": "test_resume"})
    
    similarity = embedder.compute_similarity(1, 1)
    print(f"Similarity score: {similarity:.3f}")
