"""
Simple test script for the enhanced system
"""

import os
import sys
sys.path.append('.')

from src.enhanced_db_handler import EnhancedDBHandler
from src.info_parser import InfoParser

def test_basic_functionality():
    """Test basic functionality without complex dependencies"""
    print("Testing Enhanced Resume Ranking System...")
    
    try:
        # Test database initialization
        print("1. Testing database initialization...")
        db = EnhancedDBHandler()
        print("   Database initialized successfully")
        
        # Test info parser
        print("2. Testing info parser...")
        parser = InfoParser()
        print("   Info parser initialized successfully")
        
        # Test database stats
        print("3. Testing database statistics...")
        stats = db.get_database_stats()
        print(f"   Database stats retrieved: {len(stats)} metrics")
        
        print("\nAll basic tests passed!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nSystem is ready for enhanced operations!")
    else:
        print("\nSystem needs configuration. Please check dependencies.")
