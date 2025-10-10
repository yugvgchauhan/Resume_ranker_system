"""
Enhanced Main Pipeline with Advanced Features
"""

import os
import json
import argparse
from datetime import datetime
from src.enhanced_db_handler import EnhancedDBHandler
from src.enhanced_ranker import EnhancedRanker
from src.info_parser import InfoParser
from src.embedding import Embedder


def load_text_from_file(file_path):
    """Load text from a TXT or PDF file."""
    if file_path.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def run_enhanced_pipeline(jd_folder="DATA/jobs_cleaned", resume_folder="DATA/resumes_cleaned", config_path="config.json"):
    """
    Run the enhanced resume ranking pipeline
    """
    print("Starting Enhanced Resume Ranking Pipeline...")
    print("=" * 60)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        print(f"Config file not found, using defaults")
        config = {}
    
    # Initialize enhanced components
    print("\nInitializing Enhanced Components...")
    db = EnhancedDBHandler()
    parser = InfoParser()
    embedder = Embedder()
    ranker = EnhancedRanker(config_path)
    
    print("All components initialized successfully!")
    
    # Step 1: Load Job Description
    print(f"\n📋 Step 1: Loading Job Description from {jd_folder}")
    if not os.path.exists(jd_folder) or not os.listdir(jd_folder):
        print(f"❌ No job descriptions found in {jd_folder}")
        return
    
    jd_file = os.listdir(jd_folder)[0]  # Take first JD
    jd_path = os.path.join(jd_folder, jd_file)
    jd_text = load_text_from_file(jd_path)
    
    print(f"✅ Loaded: {jd_file}")
    job_info = parser.parse_job(jd_text, jd_file)
    
    # Create enhanced session
    session_id = db.insert_session_enhanced(
        job_info["job_id"], 
        f"Enhanced_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Step 2: Load Resumes
    print(f"\n👥 Step 2: Loading Resumes from {resume_folder}")
    if not os.path.exists(resume_folder) or not os.listdir(resume_folder):
        print(f"❌ No resumes found in {resume_folder}")
        return
    
    resumes_data = []
    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)
        text = load_text_from_file(file_path)
        res_info = parser.parse_resume(text, file)
        resumes_data.append({**res_info, "text": text})
    
    print(f"✅ Loaded {len(resumes_data)} resumes")
    
    # Step 3: Process Embeddings
    print(f"\n🧠 Step 3: Processing Embeddings...")
    
    # Embed job description
    embedder.process_job(job_info["job_id"], jd_text, {**job_info, "text": jd_text})
    print(f"✅ Job description embedded")
    
    # Embed resumes
    for i, res in enumerate(resumes_data, 1):
        embedder.process_resume(res["resume_id"], res["text"], res)
        print(f"✅ Resume {i}/{len(resumes_data)} embedded")
    
    # Step 4: Enhanced Ranking
    print(f"\n🎯 Step 4: Running Enhanced Ranking Algorithm...")
    
    ranking_result = ranker.rank_session_enhanced(session_id, resumes_data)
    
    if not ranking_result:
        print("❌ Ranking failed")
        return
    
    print("✅ Enhanced ranking completed successfully!")
    
    # Step 5: Display Results
    print(f"\n🏆 Step 5: Enhanced Ranking Results")
    print("=" * 60)
    
    rankings = ranking_result['rankings']
    analytics = ranking_result['analytics']
    
    print(f"📊 Session Analytics:")
    print(f"   Total Candidates: {ranking_result['total_resumes']}")
    
    if analytics and 'score_statistics' in analytics:
        stats = analytics['score_statistics']
        print(f"   Average Score: {stats['mean']:.3f}")
        print(f"   Score Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"   Standard Deviation: {stats['std']:.3f}")
    
    print(f"\n🏆 Top 10 Candidates:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Filename':<20} {'Final Score':<12} {'Semantic':<10} {'Skills':<8} {'Experience':<10}")
    print("-" * 80)
    
    for i, result in enumerate(rankings[:10], 1):
        print(f"{i:<4} {result['filename']:<20} {result['final_score']:<12.3f} "
              f"{result['semantic_score']:<10.3f} {result['skill_score']:<8.3f} "
              f"{result['experience_score']:<10.3f}")
    
    # Step 6: Detailed Analysis
    print(f"\n📈 Step 6: Detailed Analysis")
    print("-" * 40)
    
    # Score distribution
    if analytics and 'score_distribution' in analytics:
        dist = analytics['score_distribution']
        print(f"Score Distribution:")
        print(f"   Excellent (≥0.8): {dist['excellent']} candidates")
        print(f"   Good (0.6-0.8):   {dist['good']} candidates")
        print(f"   Average (0.4-0.6): {dist['average']} candidates")
        print(f"   Poor (<0.4):      {dist['poor']} candidates")
    
    # Top skills analysis
    if analytics and 'top_skills' in analytics:
        top_skills = analytics['top_skills']
        if top_skills['most_common']:
            print(f"\nMost Common Skills:")
            for skill, count in top_skills['most_common'][:5]:
                print(f"   {skill}: {count} candidates")
    
    # Experience analysis
    if analytics and 'experience_analysis' in analytics:
        exp_analysis = analytics['experience_analysis']
        if exp_analysis:
            print(f"\nExperience Analysis:")
            print(f"   Average Experience: {exp_analysis.get('mean_experience', 0):.1f} years")
            print(f"   Median Experience: {exp_analysis.get('median_experience', 0):.1f} years")
    
    # Education analysis
    if analytics and 'education_analysis' in analytics:
        edu_analysis = analytics['education_analysis']
        if edu_analysis:
            print(f"\nEducation Distribution:")
            for level, count in edu_analysis.items():
                print(f"   {level.title()}: {count} candidates")
    
    print(f"\n✅ Enhanced Pipeline Completed Successfully!")
    print(f"📊 Session ID: {session_id}")
    print(f"🎯 Results saved to database")


def run_quick_test():
    """Run a quick test with sample data"""
    print("Running Quick Test...")
    
    # Create sample data
    sample_job = """
    Python Developer Position
    
    We are looking for an experienced Python developer to join our team.
    
    Requirements:
    - 3+ years of Python development experience
    - Experience with Django or Flask frameworks
    - Knowledge of SQL databases (PostgreSQL, MySQL)
    - Experience with REST APIs
    - Knowledge of Git version control
    - Bachelor's degree in Computer Science or related field
    
    Preferred Skills:
    - AWS or cloud platform experience
    - Docker containerization
    - CI/CD pipelines
    - Machine learning libraries (scikit-learn, pandas)
    """
    
    sample_resumes = [
        {
            "filename": "john_doe.txt",
            "text": """
            John Doe
            Senior Python Developer
            
            Experience: 5 years
            
            Skills:
            - Python (5 years)
            - Django, Flask
            - PostgreSQL, MySQL
            - REST APIs
            - Git
            - AWS
            - Docker
            
            Education: Bachelor of Computer Science
            
            Experience:
            - Senior Developer at TechCorp (2 years)
            - Python Developer at StartupXYZ (3 years)
            """
        },
        {
            "filename": "jane_smith.txt",
            "text": """
            Jane Smith
            Software Engineer
            
            Experience: 2 years
            
            Skills:
            - Python
            - JavaScript
            - React
            - MongoDB
            - Git
            
            Education: Bachelor of Engineering
            
            Experience:
            - Software Engineer at WebDev Inc (2 years)
            """
        }
    ]
    
    # Initialize components
    db = EnhancedDBHandler()
    parser = InfoParser()
    embedder = Embedder()
    ranker = EnhancedRanker()
    
    # Process job
    job_info = parser.parse_job(sample_job, "sample_job.txt")
    session_id = db.insert_session_enhanced(job_info["job_id"], "Quick_Test_Session")
    embedder.process_job(job_info["job_id"], sample_job, {**job_info, "text": sample_job})
    
    # Process resumes
    resumes_data = []
    for resume in sample_resumes:
        res_info = parser.parse_resume(resume["text"], resume["filename"])
        embedder.process_resume(res_info["resume_id"], resume["text"], {**res_info, "text": resume["text"]})
        resumes_data.append({**res_info, "text": resume["text"]})
    
    # Run ranking
    result = ranker.rank_session_enhanced(session_id, resumes_data)
    
    print("Quick test completed!")
    if result:
        print(f"📊 Ranked {result['total_resumes']} candidates")
        for i, ranking in enumerate(result['rankings'], 1):
            print(f"{i}. {ranking['filename']}: {ranking['final_score']:.3f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Resume Ranking System")
    parser.add_argument("--mode", choices=["pipeline", "test", "web"], default="pipeline",
                       help="Mode to run: pipeline (batch), test (quick test), web (streamlit)")
    parser.add_argument("--jd-folder", default="DATA/jobs_cleaned",
                       help="Folder containing job descriptions")
    parser.add_argument("--resume-folder", default="DATA/resumes_cleaned",
                       help="Folder containing resumes")
    parser.add_argument("--config", default="config.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        run_enhanced_pipeline(args.jd_folder, args.resume_folder, args.config)
    elif args.mode == "test":
        run_quick_test()
    elif args.mode == "web":
        print("🚀 Starting Enhanced Web Application...")
        print("Run: streamlit run enhanced_app.py")
    else:
        print("❌ Invalid mode specified")


if __name__ == "__main__":
    main()
