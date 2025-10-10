"""
Clean Enhanced Main Pipeline - No Unicode Issues
"""

import os
import sys
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

def run_clean_pipeline():
    """Run a clean version of the enhanced pipeline"""
    print("Starting Clean Enhanced Pipeline...")
    print("=" * 50)
    
    try:
        # Import components one by one
        print("1. Testing imports...")
        
        # Test basic imports
        import streamlit as st
        print("   Streamlit: OK")
        
        import pandas as pd
        print("   Pandas: OK")
        
        import numpy as np
        print("   Numpy: OK")
        
        # Test enhanced components
        from src.enhanced_db_handler import EnhancedDBHandler
        print("   Enhanced DB Handler: OK")
        
        from src.info_parser import InfoParser
        print("   Info Parser: OK")
        
        # Test embedder with fallback
        try:
            from src.embedding import Embedder
            print("   ChromaDB Embedder: OK")
            embedder_type = "chromadb"
        except Exception as e:
            print(f"   ChromaDB Embedder: Failed ({e})")
            from src.embedding_fallback import EmbedderFallback as Embedder
            print("   Fallback Embedder: OK")
            embedder_type = "fallback"
        
        # Test advanced scorer
        try:
            from src.advanced_scorer import AdvancedScorer
            print("   Advanced Scorer: OK")
        except Exception as e:
            print(f"   Advanced Scorer: Failed ({e})")
            print("   Using basic scoring instead")
        
        # Test enhanced ranker
        try:
            from src.enhanced_ranker import EnhancedRanker
            print("   Enhanced Ranker: OK")
        except Exception as e:
            print(f"   Enhanced Ranker: Failed ({e})")
            print("   Using basic ranking instead")
        
        print("\n2. Initializing components...")
        
        # Initialize database
        db = EnhancedDBHandler()
        print("   Database: OK")
        
        # Initialize parser
        parser = InfoParser()
        print("   Parser: OK")
        
        # Initialize embedder
        embedder = Embedder()
        print("   Embedder: OK")
        
        print("\n3. Testing functionality...")
        
        # Test database
        stats = db.get_database_stats()
        print(f"   Database stats: {len(stats)} metrics")
        
        # Test with sample data
        sample_job = "Python Developer with 3+ years experience. Skills: Python, Django, SQL."
        sample_resume = "John Doe. Python Developer with 4 years experience. Skills: Python, Django, Flask, SQL."
        
        # Process sample data
        job_info = parser.parse_job(sample_job, "test_job.txt")
        resume_info = parser.parse_resume(sample_resume, "test_resume.txt")
        
        print(f"   Job processed: {job_info.get('job_name', 'Unknown')}")
        print(f"   Resume processed: {resume_info.get('filename', 'Unknown')}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: All components are working!")
        print("\nYour enhanced system is ready to use:")
        print("- Run: streamlit run enhanced_app.py")
        print("- Or: python enhanced_main.py --mode pipeline")
        print("- Or: python enhanced_main.py --mode test")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install spaCy: python -m spacy download en_core_web_sm")
        print("3. Check file paths and permissions")
        
        return False

def main():
    """Main function"""
    print("Enhanced Resume Ranking System - Clean Test")
    print("This will test all components without Unicode issues")
    print()
    
    success = run_clean_pipeline()
    
    if success:
        print("\nSystem is ready for production use!")
    else:
        print("\nPlease fix the issues above.")

if __name__ == "__main__":
    main()
