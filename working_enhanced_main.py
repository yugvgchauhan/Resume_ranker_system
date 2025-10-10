"""
Working Enhanced Main Pipeline - Simplified Version
"""

import os
import json
from datetime import datetime

def load_text_from_file(file_path):
    """Load text from a TXT or PDF file."""
    if file_path.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def run_working_pipeline():
    """Run a working version of the enhanced pipeline"""
    print("Starting Working Enhanced Pipeline...")
    print("=" * 50)
    
    try:
        # Import components
        print("1. Importing components...")
        from src.enhanced_db_handler import EnhancedDBHandler
        from src.info_parser import InfoParser
        print("   Components imported successfully")
        
        # Initialize components
        print("2. Initializing components...")
        db = EnhancedDBHandler()
        parser = InfoParser()
        print("   Components initialized successfully")
        
        # Test with sample data
        print("3. Testing with sample data...")
        
        # Sample job description
        sample_job = """
        Python Developer Position
        
        We are looking for an experienced Python developer.
        Requirements:
        - 3+ years Python experience
        - Django/Flask frameworks
        - SQL databases
        - REST APIs
        - Git version control
        - Bachelor's degree preferred
        """
        
        # Sample resume
        sample_resume = """
        John Doe
        Senior Python Developer
        
        Experience: 4 years
        
        Skills:
        - Python (4 years)
        - Django, Flask
        - PostgreSQL, MySQL
        - REST APIs
        - Git
        - AWS
        
        Education: Bachelor of Computer Science
        
        Experience:
        - Senior Developer at TechCorp (2 years)
        - Python Developer at StartupXYZ (2 years)
        """
        
        # Process job
        print("4. Processing job description...")
        job_info = parser.parse_job(sample_job, "sample_job.txt")
        print(f"   Job processed: {job_info['job_name']}")
        
        # Process resume
        print("5. Processing resume...")
        resume_info = parser.parse_resume(sample_resume, "sample_resume.txt")
        print(f"   Resume processed: {resume_info['filename']}")
        
        # Get database stats
        print("6. Checking database...")
        stats = db.get_database_stats()
        print(f"   Database stats: {len(stats)} metrics")
        
        print("\n" + "=" * 50)
        print("SUCCESS: Working pipeline completed!")
        print("The enhanced system is ready to use.")
        print("\nNext steps:")
        print("- Run: streamlit run enhanced_app.py")
        print("- Or: python enhanced_main.py --mode pipeline")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Install spaCy model: python -m spacy download en_core_web_sm")
        print("3. Check that all files are in the correct locations")
        return False

def main():
    """Main function"""
    success = run_working_pipeline()
    
    if success:
        print("\nEnhanced system is working correctly!")
    else:
        print("\nPlease fix the issues above and try again.")

if __name__ == "__main__":
    main()
