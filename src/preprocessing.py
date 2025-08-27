# src/preprocessing.py

import os
import re
from nltk.corpus import wordnet


class Preprocessor:
    def __init__(self):
        pass  # Stopwords removal not needed for embeddings

    # ----------------- Text Cleaning -----------------
    def clean_text(self, text: str) -> str:
        # Convert spaced uppercase names: "H A R S H" → "HARSH"
        text = re.sub(r'(\b(?:[A-Z] )+[A-Z]\b)', lambda m: m.group(0).replace(' ', ''), text)

        # Remove tabs
        text = re.sub(r'\t+', ' ', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove unusual symbols except basic punctuation
        text = re.sub(r'[^\w\s\.,\-\'@]', '', text)

        # Strip each line and preserve line breaks
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

    # ----------------- Synonym Expansion (for ATS) -----------------
    def expand_with_synonyms(self, text: str) -> str:
        """
        Expand text with synonyms using WordNet.
        Example: 'developer' -> 'developer programmer'
        """
        words = text.split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)
            syns = wordnet.synsets(word)
            for syn in syns[:2]:  # limit to 2 synsets
                for lemma in syn.lemmas()[:2]:  # max 2 synonyms
                    synonym = lemma.name().replace("_", " ").lower()
                    if synonym != word and synonym not in expanded_words:
                        expanded_words.append(synonym)

        return " ".join(expanded_words)

    # ----------------- Process Folder -----------------
    def process_folder(self, input_folder: str, output_folder: str, expand_synonyms=False):
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)
        if not files:
            print(f"[INFO] No files found in {input_folder}")
            return

        for filename in files:
            file_path = os.path.join(input_folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                cleaned_text = self.clean_text(text)

                if expand_synonyms:
                    cleaned_text = self.expand_with_synonyms(cleaned_text)

                save_path = os.path.join(output_folder, filename)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                print(f"[✔] Cleaned {filename} → saved to {save_path}")
            except Exception as e:
                print(f"[❌] Failed on {filename}: {e}")


# ----------------- Quick Test -----------------
if __name__ == "__main__":
    preprocessor = Preprocessor()

    # Resumes
    resumes_processed = "data/resumes_processed"
    resumes_cleaned = "data/resumes_cleaned"
    preprocessor.process_folder(resumes_processed, resumes_cleaned, expand_synonyms=False)

    # Job Descriptions
    jobs_processed = "data/jobs_processed"
    jobs_cleaned = "data/jobs_cleaned"
    preprocessor.process_folder(jobs_processed, jobs_cleaned, expand_synonyms=False)
