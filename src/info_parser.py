import os
import re
from src.db_handler import DBHandler   # ✅ keep consistent import
# ❌ removed chromadb here (embedding handled separately)


class InfoParser:
    def __init__(self):
        self.db = DBHandler()  # initialize DBHandler
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[\s-]?)?(\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4})')
        self.experience_pattern = re.compile(r'(\d+\.?\d*)\s*(years|yrs|year|months|month)', re.I)
        self.skip_words = ['resume', 'curriculum', 'cv', 'profile', 'contact', 'address', 'about me', 'objective']

    # ----------------- Fix spaced uppercase letters -----------------
    def fix_uppercase_name(self, text):
        def repl(match):
            return ''.join(match.group(0).split())
        return re.sub(r'(?:\b[A-Z] ?){2,}', repl, text)

    # ----------------- Name Extraction -----------------
    def extract_name(self, text):
        text = self.fix_uppercase_name(text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Common location words to filter (to avoid picking locations as names)
        location_keywords = ['india', 'usa', 'canada', 'australia', 'germany', 'ahmedabad', 'mumbai', 'delhi']
        for line in lines[:15]:  # only first 15 lines likely contain name
            lower = line.lower()
            if any(word in lower for word in self.skip_words):
                continue
            if any(loc in lower for loc in location_keywords):
                continue
            if self.email_pattern.search(line) or self.phone_pattern.search(line):
                continue
            if re.search(r'\d', line):
                continue
            # Heuristics: prefer lines with 2-4 words, letters only
            tokens = [t for t in re.findall(r"[A-Za-z]+", line) if t]
            if 1 <= len(tokens) <= 4:
                return " ".join(tokens).title()
        return "Unknown"

    def extract_email(self, text):
        match = self.email_pattern.search(text)
        return match.group(0) if match else ""

    def extract_phone(self, text):
        match = self.phone_pattern.search(text)
        return match.group(0) if match else ""

    def extract_experience(self, text):
        match = self.experience_pattern.search(text)
        if match:
            exp = float(match.group(1))
            if 'month' in match.group(2).lower():
                exp = round(exp / 12, 2)
            return exp
        return 0.0

    # ----------------- Parse Resume -----------------
    def parse_resume(self, text, filename):
        data = {
            "filename": filename,
            "name": self.extract_name(text),
            "email": self.extract_email(text),
            "phone": self.extract_phone(text),
            "experience": self.extract_experience(text)
        }

        # Insert into DB (deduplication handled inside DBHandler)
        resume_id = self.db.insert_resume(
            filename=data["filename"],
            name=data["name"],
            email=data["email"],
            phone=data["phone"],
            experience=data["experience"]
        )

        # ✅ Don't add to Chroma here (embedding handled by Embedder)
        return {**data, "resume_id": resume_id}

    # ----------------- Parse Job -----------------
    def parse_job(self, text, filename):
        job_name = os.path.splitext(filename)[0]
        data = {
            "job_name": job_name,
            "exp_required": self.extract_experience(text)
        }

        # Insert into DB (job)
        job_id = self.db.insert_job(
            job_name=data["job_name"],
            exp_required=data["exp_required"]
        )

        # Create a new session for this JD
        session_id = self.db.insert_session(job_id)

        # ✅ Don't add to Chroma here (embedding handled by Embedder)
        return {**data, "job_id": job_id, "session_id": session_id}


# ----------------- Quick Test -----------------
if __name__ == "__main__":
    parser = InfoParser()

    resumes_cleaned = "DATA/resumes_cleaned"
    print("Processing Resumes...")
    if os.path.exists(resumes_cleaned):
        for filename in os.listdir(resumes_cleaned):
            filepath = os.path.join(resumes_cleaned, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            info = parser.parse_resume(text, filename)
            print(f"[Resume] {filename} → {info}")
    else:
        print(f"[INFO] Folder not found: {resumes_cleaned}")

    jobs_cleaned = "DATA/jobs_cleaned"
    print("\nProcessing Jobs...")
    if os.path.exists(jobs_cleaned):
        for filename in os.listdir(jobs_cleaned):
            filepath = os.path.join(jobs_cleaned, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            info = parser.parse_job(text, filename)
            print(f"[Job] {filename} → {info}")
    else:
        print(f"[INFO] Folder not found: {jobs_cleaned}")
