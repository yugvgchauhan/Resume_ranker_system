"""
Advanced Resume Scoring System
Implements multiple sophisticated algorithms for resume ranking
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import re
import spacy
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class AdvancedScorer:
    """
    Advanced resume scoring system with multiple sophisticated algorithms
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self._initialize_models()
        self._initialize_skills_database()
        
        # Initialize embedder (ChromaDB or fallback)
        try:
            from src.embedding import Embedder
            self.embedder = Embedder()
            self.embedder_type = "chromadb"
        except Exception as e:
            print(f"ChromaDB embedder failed: {e}")
            print("Using fallback embedder")
            from src.embedding_fallback import EmbedderFallback
            self.embedder = EmbedderFallback()
            self.embedder_type = "fallback"
        
    def _default_config(self):
        """Default configuration for scoring weights"""
        return {
            'semantic_weight': 0.25,
            'skill_match_weight': 0.25,
            'experience_weight': 0.15,
            'education_weight': 0.10,
            'keyword_weight': 0.15,
            'structure_weight': 0.05,
            'tfidf_weight': 0.05
        }
    
    def _initialize_models(self):
        """Initialize all required models"""
        print("Loading advanced NLP models...")
        
        # Sentence transformer for semantic similarity
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        
        # BERT model for contextual understanding
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0
        )
        
        # Spacy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        print("Models loaded successfully!")
    
    def _initialize_skills_database(self):
        """Initialize skills and technologies database"""
        self.skills_db = {
            'programming_languages': {
                'python': ['python', 'py'],
                'java': ['java', 'jdk', 'jre'],
                'javascript': ['javascript', 'js', 'es6', 'es7', 'es2015'],
                'typescript': ['typescript', 'ts'],
                'c++': ['c++', 'cpp', 'c plus plus'],
                'c#': ['c#', 'csharp', '.net', 'dotnet', 'asp.net', 'aspnet'],
                'go': ['go', 'golang'],
                'rust': ['rust'],
                'php': ['php'],
                'ruby': ['ruby'],
                'swift': ['swift'],
                'kotlin': ['kotlin'],
                'scala': ['scala'],
                'r': ['r language', 'r (programming language)', 'r'],
                'sql': ['sql']
            },
            'frontend_frameworks': {
                'react': ['react', 'reactjs', 'react.js', 'nextjs', 'next.js'],
                'angular': ['angular', 'angularjs', 'angular.js'],
                'vue': ['vue', 'vuejs', 'vue.js', 'nuxt', 'nuxt.js'],
                'svelte': ['svelte', 'sveltekit']
            },
            'backend_frameworks': {
                'nodejs': ['node', 'nodejs', 'node.js', 'express', 'express.js', 'nest', 'nestjs'],
                'django': ['django', 'django rest framework', 'drf'],
                'flask': ['flask'],
                'fastapi': ['fastapi'],
                'spring': ['spring', 'spring boot', 'springboot'],
                'rails': ['rails', 'ruby on rails'],
                'laravel': ['laravel'],
                'dotnet': ['.net', 'asp.net', 'aspnet', 'entity framework']
            },
            'mobile': {
                'android': ['android', 'kotlin', 'java (android)'],
                'ios': ['ios', 'swift', 'objective-c', 'objective c'],
                'react_native': ['react native'],
                'flutter': ['flutter', 'dart']
            },
            'databases': {
                'mysql': ['mysql'],
                'postgresql': ['postgresql', 'postgres', 'psql'],
                'sqlite': ['sqlite'],
                'oracle': ['oracle', 'pl/sql', 'plsql'],
                'mssql': ['mssql', 'sql server', 'microsoft sql server'],
                'mongodb': ['mongodb', 'mongo'],
                'cassandra': ['cassandra'],
                'redis': ['redis'],
                'elasticsearch': ['elasticsearch', 'elastic search', 'elk']
            },
            'data_engineering': {
                'spark': ['spark', 'pyspark', 'apache spark'],
                'hadoop': ['hadoop', 'mapreduce', 'hdfs'],
                'airflow': ['airflow', 'apache airflow'],
                'kafka': ['kafka', 'apache kafka'],
                'dbt': ['dbt', 'data build tool']
            },
            'ml_ai': {
                'tensorflow': ['tensorflow', 'tf'],
                'pytorch': ['pytorch', 'torch'],
                'scikit-learn': ['scikit-learn', 'sklearn'],
                'keras': ['keras'],
                'xgboost': ['xgboost'],
                'lightgbm': ['lightgbm', 'lgbm'],
                'opencv': ['opencv', 'cv2'],
                'nlp': ['nlp', 'spacy', 'nltk', 'transformers', 'hugging face', 'huggingface'],
                'llm': ['llm', 'gpt', 'openai', 'langchain']
            },
            'cloud_platforms': {
                'aws': ['aws', 'amazon web services', 's3', 'ec2', 'lambda', 'rds', 'cloudwatch', 'iam', 'ecs', 'eks'],
                'azure': ['azure', 'microsoft azure', 'azure devops', 'aks', 'cosmos db', 'azure functions'],
                'gcp': ['gcp', 'google cloud', 'google cloud platform', 'bigquery', 'cloud run', 'gke']
            },
            'devops_tools': {
                'docker': ['docker'],
                'kubernetes': ['kubernetes', 'k8s'],
                'terraform': ['terraform'],
                'ansible': ['ansible'],
                'jenkins': ['jenkins'],
                'github_actions': ['github actions', 'gh actions'],
                'gitlab_ci': ['gitlab ci', 'gitlab-ci'],
                'circleci': ['circleci']
            },
            'data_tools': {
                'pandas': ['pandas'],
                'numpy': ['numpy', 'np'],
                'matplotlib': ['matplotlib', 'mpl'],
                'seaborn': ['seaborn'],
                'plotly': ['plotly'],
                'power_bi': ['powerbi', 'power bi'],
                'tableau': ['tableau']
            },
            'web_apis': {
                'rest': ['rest', 'restful', 'rest api', 'restful api'],
                'graphql': ['graphql', 'graph ql'],
                'grpc': ['grpc']
            },
            'testing': {
                'pytest': ['pytest', 'py.test'],
                'unittest': ['unittest'],
                'jest': ['jest'],
                'cypress': ['cypress'],
                'playwright': ['playwright']
            },
            'version_control': {
                'git': ['git', 'github', 'gitlab', 'bitbucket']
            },
            'methodologies': {
                'agile': ['agile', 'scrum', 'kanban'],
                'oop': ['oop', 'object oriented programming', 'object-oriented programming'],
                'dsa': ['dsa', 'data structures', 'algorithms']
            }
        }
        
        # Education levels with weights
        self.education_weights = {
            'phd': 1.0,
            'master': 0.8,
            'bachelor': 0.6,
            'diploma': 0.4,
            'certification': 0.3,
            'high_school': 0.2
        }
    
    def extract_skills(self, text):
        """Section-aware skill extraction (preferred earlier), returning only curated skills.

        - Locate Skills/Tech Stack section and parse tokens
        - Classify tokens against curated skills DB with word boundaries
        - Do NOT return free-form 'other_skills' (per user's request)
        - If no section found, fall back to global matching
        """
        heading_pat = r"(?mi)^(skills|technical skills|key skills|skills summary|skills & tools|technical proficiencies|tech stack|core competencies)\s*[:\-]?\s*$"
        next_heading_pat = r"(?mi)^(experience|work experience|projects|education|summary|profile|certifications|achievements|internships|languages)\b|^\s*[A-Z][A-Z &/]{2,}\s*$"

        sections = []
        for m in re.finditer(heading_pat, text):
            start = m.end()
            next_m = re.search(next_heading_pat, text[start:])
            end = start + next_m.start() if next_m else len(text)
            sections.append(text[start:end])

        skill_blob = "\n".join(sections) if sections else text
        parts = re.split(r"[\n,;\|/•·\u2022\t]+", skill_blob)

        def _normalize_token(token: str) -> str:
            t = token.strip()
            # collapse multiple spaces and remove common wrappers
            t = re.sub(r"\s+", " ", t)
            t = t.strip("-•·|,;:/\\")
            # lowercase, normalize dots in tech names (e.g., Node.js -> nodejs)
            tl = t.lower()
            tl = tl.replace(".js", "js").replace(".net", " net ")
            # remove parentheses content
            tl = re.sub(r"\([^)]*\)", "", tl).strip()
            return tl if tl else ""

        raw_tokens = [p for p in parts if p.strip()]
        tokens = []
        for p in raw_tokens:
            # split on common joiners within a bullet
            subparts = re.split(r"\s*[•·\u2022:/,;|]+\s*", p)
            for sp in subparts:
                norm = _normalize_token(sp)
                if norm:
                    tokens.append(norm)

        # Filter obvious non-skill noise
        generic = set(['skills','technical','tools','technology','proficiencies','and','with','proficient','familiar','contact','profile','summary','education','certificates','internships','languages'])
        email_re = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-z]{2,}")
        url_re = re.compile(r"https?://|www\.", re.I)
        phone_re = re.compile(r"\b\+?\d[\d\s\-()]{7,}\b")
        percent_re = re.compile(r"\b\d{1,3}\.?\d*%\b")

        filtered = []
        seen = set()
        for tok in tokens:
            lt = tok.strip(' .')
            if not lt or lt in generic or lt in self.stop_words:
                continue
            if email_re.search(lt) or url_re.search(lt) or phone_re.search(lt) or percent_re.search(lt):
                continue
            if len(lt) > 50 or re.search(r"\d{3,}", lt):
                continue
            if lt not in seen:
                seen.add(lt)
                filtered.append(lt)

        # Classify against curated DB
        found_skills = {}
        text_joined = " \n ".join(filtered)
        for category, skills in self.skills_db.items():
            hits = set()
            for skill, variations in skills.items():
                for variation in variations:
                    # allow word-boundary or exact token match after normalization
                    if re.search(r"(?i)\b" + re.escape(variation) + r"\b", text_joined):
                        hits.add(skill)
                        break
            found_skills[category] = sorted(hits)

        # Do not include other_skills
        found_skills['other_skills'] = []
        return found_skills
    
    def calculate_skill_similarity(self, job_skills, resume_skills):
        """Calculate skill matching score between job and resume"""
        if not job_skills or not resume_skills:
            return 0.0
        
        total_skills = 0
        matched_skills = 0
        
        for category in job_skills:
            if category in resume_skills:
                job_cat_skills = set(job_skills[category])
                resume_cat_skills = set(resume_skills[category])
                
                total_skills += len(job_cat_skills)
                matched_skills += len(job_cat_skills.intersection(resume_cat_skills))
        
        return matched_skills / total_skills if total_skills > 0 else 0.0
    
    def extract_experience_details(self, text):
        """Extract detailed experience information"""
        # Years of experience
        exp_pattern = r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|months?)'
        exp_matches = re.findall(exp_pattern, text.lower())
        
        total_exp = 0
        for match in exp_matches:
            exp = float(match)
            if 'month' in text.lower():
                exp = exp / 12
            total_exp += exp
        
        # Job titles and companies
        job_patterns = [
            r'(?:worked at|experience at|employed at)\s+([A-Z][a-zA-Z\s&]+)',
            r'(?:software engineer|developer|analyst|manager|director|lead|senior|junior)',
            r'(?:intern|internship|trainee|associate)'
        ]
        
        job_titles = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_titles.extend(matches)
        
        return {
            'total_years': total_exp,
            'job_titles': job_titles,
            'has_internship': 'intern' in text.lower()
        }
    
    def extract_education(self, text):
        """Extract highest education level using targeted degree patterns"""
        txt = text.lower()
        patterns = {
            'phd': [r'ph\.?d\.?', r'doctorate', r'doctoral'],
            'master': [r"master of\b", r"\bm\.?s\.?\b", r"\bm\.?tech\.?\b", r"\bmca\b", r"\bmba\b"],
            'bachelor': [r"bachelor\b", r"b\.?e\.?\b", r"b\.?tech\.?\b", r"bsc\b", r"b\.a\b", r"bca\b", r"bcom\b"],
            'diploma': [r"diploma\b", r"pgdm\b"],
            'certification': [r"certificate\b", r"certified\b", r"certification\b"]
        }
        # Prefer bachelor if both appear but master's not explicit 'Master of' etc.
        # Prefer bachelor matches over master when both appear
        for level in ['phd', 'bachelor', 'master', 'diploma', 'certification']:
            for pat in patterns[level]:
                if re.search(pat, txt):
                    return level
        return 'high_school'
    
    def calculate_semantic_similarity_advanced(self, job_text, resume_text):
        """Advanced semantic similarity using multiple models"""
        try:
            # Sentence transformer similarity
            job_embedding = self.semantic_model.encode(job_text)
            resume_embedding = self.semantic_model.encode(resume_text)
            semantic_score = cosine_similarity(
                job_embedding.reshape(1, -1), 
                resume_embedding.reshape(1, -1)
            )[0][0]
            
            # BERT-based contextual similarity (optional, can be slow)
            try:
                bert_score = self._calculate_bert_similarity(job_text, resume_text)
                # Combine both scores
                return (semantic_score + bert_score) / 2
            except Exception as e:
                print(f"BERT similarity failed, using sentence transformer only: {e}")
                return semantic_score
            
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0
    
    def _calculate_bert_similarity(self, text1, text2):
        """Calculate BERT-based similarity"""
        try:
            # Tokenize and encode
            tokens1 = self.bert_tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
            tokens2 = self.bert_tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs1 = self.bert_model(**tokens1)
                outputs2 = self.bert_model(**tokens2)
                
                # Use [CLS] token embedding
                embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
                embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error in BERT similarity: {e}")
            return 0.0
    
    def calculate_tfidf_similarity(self, job_text, resume_text):
        """Calculate TF-IDF based similarity"""
        try:
            # Guard for tiny inputs
            docs = [job_text or "", resume_text or ""]
            if sum(len(d.strip()) > 0 for d in docs) < 2:
                return 0.0
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error in TF-IDF similarity: {e}")
            return 0.0
    
    def calculate_experience_score(self, job_exp_required, resume_exp):
        """Calculate experience matching score"""
        if job_exp_required == 0:
            return 1.0 if resume_exp > 0 else 0.5
        
        # Penalize over-qualification slightly
        if resume_exp >= job_exp_required * 2:
            return 0.8
        
        # Reward exact or close match
        ratio = resume_exp / job_exp_required
        if ratio >= 0.8 and ratio <= 1.2:
            return 1.0
        elif ratio >= 0.5:
            return 0.8
        else:
            return ratio
    
    def calculate_education_score(self, job_text, resume_education):
        """Calculate education matching score"""
        # Extract education requirements from job
        job_education = self.extract_education(job_text)
        
        # If no specific education mentioned, give neutral score
        if job_education == 'high_school':
            return 0.7
        
        # Calculate score based on education levels
        job_weight = self.education_weights.get(job_education, 0.5)
        resume_weight = self.education_weights.get(resume_education, 0.5)
        
        # Reward higher education
        if resume_weight >= job_weight:
            return 1.0
        else:
            return resume_weight / job_weight
    
    def calculate_structure_score(self, resume_text):
        """Calculate resume structure and formatting score"""
        score = 0.0
        
        # Check for essential sections
        sections = ['experience', 'education', 'skills', 'contact', 'summary']
        found_sections = sum(1 for section in sections if section in resume_text.lower())
        score += (found_sections / len(sections)) * 0.4
        
        # Check for professional formatting
        if len(resume_text.split('\n')) > 10:  # Has structure
            score += 0.2
        
        if '@' in resume_text and re.search(r'\d{10}', resume_text):  # Has contact info
            score += 0.2
        
        if len(resume_text.split()) > 100:  # Sufficient content
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_comprehensive_score(self, job_text, resume_text, resume_metadata=None):
        """Calculate comprehensive score using all metrics"""
        # Extract information
        job_skills = self.extract_skills(job_text)
        resume_skills = self.extract_skills(resume_text)
        job_exp = self.extract_experience_details(job_text)
        resume_exp = self.extract_experience_details(resume_text)
        resume_education = self.extract_education(resume_text)
        
        # Calculate individual scores
        semantic_score = self.calculate_semantic_similarity_advanced(job_text, resume_text)
        skill_score = self.calculate_skill_similarity(job_skills, resume_skills)
        experience_score = self.calculate_experience_score(job_exp['total_years'], resume_exp['total_years'])
        education_score = self.calculate_education_score(job_text, resume_education)
        structure_score = self.calculate_structure_score(resume_text)
        tfidf_score = self.calculate_tfidf_similarity(job_text, resume_text)
        
        # Keyword overlap (existing method)
        keyword_score = self._calculate_keyword_overlap(job_text, resume_text)
        
        # Weighted combination
        weights = self.config
        final_score = (
            weights['semantic_weight'] * semantic_score +
            weights['skill_match_weight'] * skill_score +
            weights['experience_weight'] * experience_score +
            weights['education_weight'] * education_score +
            weights['structure_weight'] * structure_score +
            weights['tfidf_weight'] * tfidf_score +
            weights['keyword_weight'] * keyword_score
        )
        
        return {
            'final_score': final_score,
            'breakdown': {
                'semantic_similarity': semantic_score,
                'skill_match': skill_score,
                'experience_match': experience_score,
                'education_match': education_score,
                'structure_score': structure_score,
                'tfidf_similarity': tfidf_score,
                'keyword_overlap': keyword_score
            },
            'extracted_info': {
                'job_skills': job_skills,
                'resume_skills': resume_skills,
                'job_experience': job_exp,
                'resume_experience': resume_exp,
                'resume_education': resume_education
            }
        }
    
    def _calculate_keyword_overlap(self, job_text, resume_text):
        """Calculate keyword overlap score"""
        job_words = set(re.findall(r'\b\w+\b', job_text.lower()))
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        
        # Remove stop words
        job_words = job_words - self.stop_words
        resume_words = resume_words - self.stop_words
        
        if not job_words:
            return 0.0
        
        overlap = len(job_words.intersection(resume_words))
        return overlap / len(job_words)


# Test function
if __name__ == "__main__":
    scorer = AdvancedScorer()
    
    # Sample texts
    job_text = """
    We are looking for a Python Developer with 3+ years of experience.
    Skills required: Python, Django, SQL, AWS, Git.
    Bachelor's degree in Computer Science preferred.
    """
    
    resume_text = """
    John Doe
    Software Engineer with 4 years of experience in Python development.
    Skills: Python, Django, Flask, PostgreSQL, AWS, Docker, Git.
    Education: Bachelor of Technology in Computer Science.
    Experience: 2 years at TechCorp, 2 years at StartupXYZ.
    """
    
    result = scorer.calculate_comprehensive_score(job_text, resume_text)
    print("Advanced Scoring Result:")
    print(f"Final Score: {result['final_score']:.3f}")
    print("\nScore Breakdown:")
    for metric, score in result['breakdown'].items():
        print(f"  {metric}: {score:.3f}")
