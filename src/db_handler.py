import sqlite3
import os
from datetime import datetime

# Keep legacy path to avoid breaking older flows
DB_PATH = "databases/resume_rank.db"
os.makedirs("databases", exist_ok=True)

class DBHandler:
    def __init__(self):
        self.create_tables()

    def get_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---------------- Create Tables ----------------
    def create_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                experience REAL,
                upload_date TEXT
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT,
                exp_required REAL,
                upload_date TEXT
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                session_name TEXT,
                created_at TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                job_id INTEGER,
                resume_id INTEGER,
                cosine_score REAL,
                ats_score REAL,
                exp_score REAL,
                final_score REAL,
                rank INTEGER,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
                FOREIGN KEY (resume_id) REFERENCES resumes(resume_id) ON DELETE CASCADE,
                UNIQUE(session_id, resume_id)
            )""")
            conn.commit()
        print(f"[OK] Database initialized at {DB_PATH}")
        # Run light migrations to avoid schema errors between old/new flows
        self._run_migrations()

    def _run_migrations(self):
        """Apply non-destructive schema migrations (idempotent)."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # 1) Ensure sessions.session_name exists
                cursor.execute("PRAGMA table_info(sessions)")
                cols = [row[1] for row in cursor.fetchall()]
                if "session_name" not in cols:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN session_name TEXT")
                    conn.commit()
        except Exception as e:
            # Do not crash app if migration fails; just log
            print(f"[WARN] Migration skipped or failed: {e}")

    # ---------------- Insert ----------------
    def insert_resume(self, filename, name, email, phone, experience=0.0):
        """
        Insert a resume into DB.
        - If candidate already exists (by email or phone), update their record (latest version).
        - Otherwise, insert a new row.
        """
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # ✅ Check for existing resume by email/phone
            cursor.execute("""
                SELECT resume_id FROM resumes 
                WHERE (email != '' AND email = ?) OR (phone != '' AND phone = ?)
                ORDER BY upload_date DESC
                LIMIT 1
            """, (email, phone))
            existing = cursor.fetchone()

            if existing:
                resume_id = existing[0]
                # Update latest record with new filename & info
                cursor.execute("""
                    UPDATE resumes
                    SET filename=?, name=?, experience=?, upload_date=?
                    WHERE resume_id=?
                """, (filename, name, experience, upload_date, resume_id))
                return resume_id
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO resumes (filename, name, email, phone, experience, upload_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (filename, name, email, phone, experience, upload_date))
                return cursor.lastrowid

    def insert_job(self, job_name, exp_required=0.0):
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (job_name, exp_required, upload_date)
                VALUES (?, ?, ?)
            """, (job_name, exp_required, upload_date))
            return cursor.lastrowid

    def insert_session(self, job_id, session_name=None):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if session_name is None:
            session_name = f"Session_{created_at.replace(':', '-').replace(' ', '_')}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (job_id, session_name, created_at)
                VALUES (?, ?, ?)
            """, (job_id, session_name, created_at))
            return cursor.lastrowid

    def insert_ranking(self, session_id, job_id, resume_id, cosine_score, ats_score, exp_score, final_score, rank):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO rankings 
                (session_id, job_id, resume_id, cosine_score, ats_score, exp_score, final_score, rank, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, job_id, resume_id, cosine_score, ats_score, exp_score, final_score, rank, created_at))
            return cursor.lastrowid

    def store_rankings(self, session_id, ranked_list):
        """
        Store ranked resumes for a given session.
        ranked_list: [(resume_id, final_score), ...] sorted by score desc
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for rank, (resume_id, final_score) in enumerate(ranked_list, 1):
                cursor.execute("""
                    INSERT OR REPLACE INTO rankings
                    (session_id, job_id, resume_id, final_score, rank, created_at)
                    VALUES (?, 
                            (SELECT job_id FROM sessions WHERE session_id = ?), 
                            ?, ?, ?, datetime('now'))
                """, (session_id, session_id, resume_id, final_score, rank))
            conn.commit()


    # ---------------- Fetch ----------------
    def get_job_by_session(self, session_id):
        query = """
        SELECT j.job_id, j.job_name, j.exp_required
        FROM jobs j
        JOIN sessions s ON j.job_id = s.job_id
        WHERE s.session_id = ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()
            return {"job_id": row[0], "job_name": row[1], "exp_required": row[2]} if row else None

    def get_ranked_resumes(self, session_id, top_k=10):
        query = """
        SELECT r.filename, r.experience, r.email, r.phone, ra.final_score
        FROM rankings ra
        JOIN resumes r ON ra.resume_id = r.resume_id
        WHERE ra.session_id = ?
        GROUP BY ra.resume_id
        ORDER BY ra.final_score DESC
        LIMIT ?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id, top_k))
            rows = cursor.fetchall()
            return [{"filename": r[0], "experience": r[1], "email": r[2], "phone": r[3], "score": r[4]} for r in rows]

    def clear_session_rankings(self, session_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rankings WHERE session_id = ?", (session_id,))
            conn.commit()


# ---------------- Quick Test ----------------
if __name__ == "__main__":
    db = DBHandler()

    print("\n[DB TEST] Checking database contents...")
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables:", [t[0] for t in tables])

        # Count resumes
        cursor.execute("SELECT COUNT(*) FROM resumes;")
        print("Total resumes:", cursor.fetchone()[0])

        # Count jobs
        cursor.execute("SELECT COUNT(*) FROM jobs;")
        print("Total jobs:", cursor.fetchone()[0])

        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions;")
        print("Total sessions:", cursor.fetchone()[0])

        # Count rankings
        cursor.execute("SELECT COUNT(*) FROM rankings;")
        print("Total rankings:", cursor.fetchone()[0])
