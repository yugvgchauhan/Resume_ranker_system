# src/extractor.py

import os
import pdfplumber
from docx import Document

# ----------------- PDF Extraction -----------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n"  # Preserve line breaks
    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
    return text

# ----------------- DOCX Extraction -----------------
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading DOCX {file_path}: {e}")
    return text

# ----------------- TXT Extraction -----------------
def extract_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"❌ Error reading TXT {file_path}: {e}")
    return text

# ----------------- Master Extraction Function -----------------
def extract_text(file_path):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ----------------- Process Folder -----------------
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    if not files:
        print(f"[INFO] No files found in {input_folder}")
        return

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            text = extract_text(file_path)
            save_path = os.path.join(
                output_folder,
                filename.replace(".pdf", ".txt").replace(".docx", ".txt")
            )
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[✔] Processed {filename} → saved to {save_path}")
        except Exception as e:
            print(f"[❌] Failed on {filename}: {e}")

# ----------------- Quick Test -----------------
if __name__ == "__main__":
    # Resumes
    resumes_folder = "data/resumes"
    resumes_processed = "data/resumes_processed"
    process_folder(resumes_folder, resumes_processed)

    # Job Descriptions
    jobs_folder = "data/JDs"
    jobs_processed = "data/jobs_processed"
    process_folder(jobs_folder, jobs_processed)
