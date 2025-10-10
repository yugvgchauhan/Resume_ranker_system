"""
Enhanced Database Handler with improved ChromaDB and SQLite integration
"""

import sqlite3
import os
import chromadb
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


class EnhancedDBHandler:
    """
    Enhanced database handler with improved integration between SQLite and ChromaDB
    """
    
    def __init__(self, db_path="databases/resume_rank_enhanced.db", chroma_path="databases/chroma_db"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._ensure_directories()
        self._initialize_databases()
    
    def _ensure_directories(self):
        """Ensure database directories exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)
    
    def _initialize_databases(self):
        """Initialize both SQLite and ChromaDB"""
        self._create_sqlite_tables()
        self._initialize_chromadb()
        print("Enhanced database initialization completed")
    
    def _create_sqlite_tables(self):
        """Create enhanced SQLite tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enhanced resumes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    experience REAL DEFAULT 0.0,
                    education_level TEXT DEFAULT 'unknown',
                    skills_extracted TEXT,  -- JSON string
                    text_hash TEXT,  -- For deduplication
                    upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_processed TEXT,
                    UNIQUE(text_hash)
                )
            """)
            
            # Enhanced jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    exp_required REAL DEFAULT 0.0,
                    skills_required TEXT,  -- JSON string
                    education_required TEXT DEFAULT 'unknown',
                    text_hash TEXT,
                    upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_processed TEXT,
                    UNIQUE(text_hash)
                )
            """)
            
            # Enhanced sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    session_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    total_resumes INTEGER DEFAULT 0,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)
            
            # Enhanced rankings table with detailed scores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    job_id INTEGER NOT NULL,
                    resume_id INTEGER NOT NULL,
                    semantic_score REAL DEFAULT 0.0,
                    skill_score REAL DEFAULT 0.0,
                    experience_score REAL DEFAULT 0.0,
                    education_score REAL DEFAULT 0.0,
                    structure_score REAL DEFAULT 0.0,
                    tfidf_score REAL DEFAULT 0.0,
                    keyword_score REAL DEFAULT 0.0,
                    final_score REAL DEFAULT 0.0,
                    rank_position INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
                    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id) ON DELETE CASCADE,
                    UNIQUE(session_id, resume_id)
                )
            """)
            
            # Skills database table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skills_database (
                    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    synonyms TEXT,  -- JSON string
                    popularity_score REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(skill_name, category)
                )
            """)
            
            # Analytics table for tracking performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with enhanced configuration"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create collections with metadata schema
            self.resume_collection = self.chroma_client.get_or_create_collection(
                name="resumes",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.job_collection = self.chroma_client.get_or_create_collection(
                name="jobs",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.skill_collection = self.chroma_client.get_or_create_collection(
                name="skills",
                metadata={"hnsw:space": "cosine"}
            )
            
            print("ChromaDB collections initialized")
            
        except Exception as e:
            print(f"ChromaDB initialization failed: {e}")
            raise
    
    def get_connection(self):
        """Get SQLite connection with optimizations"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        return conn
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def insert_resume_enhanced(self, filename: str, text: str, metadata: Dict) -> int:
        """Insert resume with enhanced metadata and deduplication"""
        text_hash = self._generate_text_hash(text)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check for existing resume by hash
            cursor.execute("SELECT resume_id FROM resumes WHERE text_hash = ?", (text_hash,))
            existing = cursor.fetchone()
            
            if existing:
                resume_id = existing[0]
                # Update existing record
                cursor.execute("""
                    UPDATE resumes 
                    SET filename=?, name=?, email=?, phone=?, experience=?,
                        education_level=?, skills_extracted=?, last_processed=?
                    WHERE resume_id=?
                """, (
                    filename, metadata.get('name'), metadata.get('email'),
                    metadata.get('phone'), metadata.get('experience'),
                    metadata.get('education_level'), json.dumps(metadata.get('skills', {})),
                    datetime.now().isoformat(), resume_id
                ))
                return resume_id
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO resumes 
                    (filename, name, email, phone, experience, education_level, 
                     skills_extracted, text_hash, last_processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename, metadata.get('name'), metadata.get('email'),
                    metadata.get('phone'), metadata.get('experience'),
                    metadata.get('education_level'), json.dumps(metadata.get('skills', {})),
                    text_hash, datetime.now().isoformat()
                ))
                return cursor.lastrowid
    
    def insert_job_enhanced(self, job_name: str, text: str, metadata: Dict) -> int:
        """Insert job with enhanced metadata"""
        text_hash = self._generate_text_hash(text)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check for existing job by hash to avoid UNIQUE constraint error
            cursor.execute("SELECT job_id FROM jobs WHERE text_hash = ?", (text_hash,))
            existing = cursor.fetchone()
            if existing:
                job_id = existing[0]
                # Optionally update metadata on re-insert
                cursor.execute("""
                    UPDATE jobs
                    SET job_name = ?, exp_required = ?, skills_required = ?, education_required = ?,
                        last_processed = ?
                    WHERE job_id = ?
                """, (
                    job_name, metadata.get('exp_required', 0.0),
                    json.dumps(metadata.get('skills_required', {})),
                    metadata.get('education_required', 'unknown'),
                    datetime.now().isoformat(), job_id
                ))
                return job_id
            else:
                cursor.execute("""
                    INSERT INTO jobs 
                    (job_name, exp_required, skills_required, education_required, 
                     text_hash, last_processed)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    job_name, metadata.get('exp_required', 0.0),
                    json.dumps(metadata.get('skills_required', {})),
                    metadata.get('education_required', 'unknown'),
                    text_hash, datetime.now().isoformat()
                ))
                return cursor.lastrowid
    
    def insert_session_enhanced(self, job_id: int, session_name: str = None) -> int:
        """Insert session with enhanced tracking"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (job_id, session_name, status)
                VALUES (?, ?, ?)
            """, (job_id, session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 'active'))
            return cursor.lastrowid
    
    def store_rankings_enhanced(self, session_id: int, rankings_data: List[Dict]):
        """Store enhanced rankings with detailed score breakdown"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing rankings for this session
            cursor.execute("DELETE FROM rankings WHERE session_id = ?", (session_id,))
            
            # Insert new rankings
            for rank_data in rankings_data:
                # Ensure numeric fields are stored as REALs
                sem = float(rank_data.get('semantic_score', 0.0) or 0.0)
                skl = float(rank_data.get('skill_score', 0.0) or 0.0)
                exp = float(rank_data.get('experience_score', 0.0) or 0.0)
                edu = float(rank_data.get('education_score', 0.0) or 0.0)
                strc = float(rank_data.get('structure_score', 0.0) or 0.0)
                tfidf = float(rank_data.get('tfidf_score', 0.0) or 0.0)
                keyw = float(rank_data.get('keyword_score', 0.0) or 0.0)
                final = float(rank_data.get('final_score', 0.0) or 0.0)
                rank_pos = int(rank_data.get('rank_position', 0) or 0)
                cursor.execute("""
                    INSERT INTO rankings 
                    (session_id, job_id, resume_id, semantic_score, skill_score,
                     experience_score, education_score, structure_score, tfidf_score,
                     keyword_score, final_score, rank_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, rank_data['job_id'], rank_data['resume_id'],
                    sem, skl, exp, edu, strc, tfidf, keyw, final, rank_pos
                ))
            
            # Update session statistics
            cursor.execute("""
                UPDATE sessions 
                SET total_resumes = ? 
                WHERE session_id = ?
            """, (len(rankings_data), session_id))
            
            conn.commit()
    
    def get_ranked_resumes_enhanced(self, session_id: int, top_k: int = 10) -> List[Dict]:
        """Get enhanced ranked resumes with detailed information"""
        query = """
            SELECT 
                r.resume_id, r.filename, r.name, r.email, r.phone, 
                r.experience, r.education_level, r.skills_extracted,
                ra.semantic_score, ra.skill_score, ra.experience_score,
                ra.education_score, ra.structure_score, ra.tfidf_score,
                ra.keyword_score, ra.final_score, ra.rank_position
            FROM rankings ra
            JOIN resumes r ON ra.resume_id = r.resume_id
            WHERE ra.session_id = ?
            ORDER BY ra.final_score DESC, ra.rank_position ASC
            LIMIT ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id, top_k))
            rows = cursor.fetchall()
            
            def _to_float(x):
                if isinstance(x, (bytes, bytearray)):
                    try:
                        x = x.decode('utf-8', errors='ignore')
                    except Exception:
                        return 0.0
                try:
                    return float(x)
                except Exception:
                    return 0.0

            results = []
            for row in rows:
                results.append({
                    'resume_id': row[0],
                    'filename': row[1],
                    'name': row[2],
                    'email': row[3],
                    'phone': row[4],
                    'experience': _to_float(row[5]),
                    'education_level': row[6],
                    'skills_extracted': json.loads(row[7]) if row[7] else {},
                    'semantic_score': _to_float(row[8]),
                    'skill_score': _to_float(row[9]),
                    'experience_score': _to_float(row[10]),
                    'education_score': _to_float(row[11]),
                    'structure_score': _to_float(row[12]),
                    'tfidf_score': _to_float(row[13]),
                    'keyword_score': _to_float(row[14]),
                    'final_score': _to_float(row[15]),
                    'rank_position': row[16]
                })
            
            return results
    
    def get_job_by_session(self, session_id: int) -> Dict:
        """Get job information by session ID"""
        query = """
        SELECT j.job_id, j.job_name, j.exp_required, j.skills_required, j.education_required
        FROM jobs j
        JOIN sessions s ON j.job_id = s.job_id
        WHERE s.session_id = ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()
            return {
                "job_id": row[0], 
                "job_name": row[1], 
                "exp_required": row[2],
                "skills_required": json.loads(row[3]) if row[3] else {},
                "education_required": row[4]
            } if row else None

    def get_session_analytics(self, session_id: int) -> Dict:
        """Get analytics for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic session info
            cursor.execute("""
                SELECT s.session_name, s.created_at, s.total_resumes,
                       j.job_name, j.exp_required
                FROM sessions s
                JOIN jobs j ON s.job_id = j.job_id
                WHERE s.session_id = ?
            """, (session_id,))
            
            session_info = cursor.fetchone()
            if not session_info:
                return {}
            
            # Score statistics
            cursor.execute("""
                SELECT 
                    AVG(CAST(final_score AS REAL)) as avg_score,
                    MIN(CAST(final_score AS REAL)) as min_score,
                    MAX(CAST(final_score AS REAL)) as max_score,
                    COUNT(*) as total_candidates
                FROM rankings 
                WHERE session_id = ?
            """, (session_id,))
            
            stats = cursor.fetchone()
            
            return {
                'session_name': session_info[0],
                'created_at': session_info[1],
                'total_resumes': int(session_info[2] or 0),
                'job_name': session_info[3],
                'exp_required': float(session_info[4] or 0.0),
                'score_statistics': {
                    'average_score': float(stats[0] or 0.0),
                    'min_score': float(stats[1] or 0.0),
                    'max_score': float(stats[2] or 0.0),
                    'total_candidates': int(stats[3] or 0)
                }
            }
    
    def search_similar_resumes(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar resumes using ChromaDB"""
        try:
            # This would require the embedding model to be loaded
            # For now, return empty list
            return []
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['resumes', 'jobs', 'sessions', 'rankings', 'skills_database']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # ChromaDB stats
            try:
                stats['chroma_resumes_count'] = self.resume_collection.count()
                stats['chroma_jobs_count'] = self.job_collection.count()
                stats['chroma_skills_count'] = self.skill_collection.count()
            except:
                stats['chroma_resumes_count'] = 0
                stats['chroma_jobs_count'] = 0
                stats['chroma_skills_count'] = 0
            
            return stats
    
    def clear_all_data(self):
        """Clear all data from both databases"""
        # Clear SQLite
        with self.get_connection() as conn:
            cursor = conn.cursor()
            tables = ['rankings', 'sessions', 'resumes', 'jobs', 'skills_database', 'analytics']
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
            conn.commit()
        
        # Clear ChromaDB
        try:
            self.chroma_client.reset()
            self._initialize_chromadb()
        except Exception as e:
            print(f"ChromaDB reset error: {e}")
        
        print("All data cleared")


# Test function
if __name__ == "__main__":
    db = EnhancedDBHandler()
    
    print("Database Statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nDatabase initialization completed successfully!")
