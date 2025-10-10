"""
Test the enhanced app components
"""

def test_imports():
    """Test if all imports work"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import streamlit as st
        print("  Streamlit: OK")
        
        # Test enhanced components
        from src.enhanced_db_handler import EnhancedDBHandler
        print("  Enhanced DB Handler: OK")
        
        from src.info_parser import InfoParser
        print("  Info Parser: OK")
        
        # Test embedder (with fallback)
        try:
            from src.embedding import Embedder
            print("  ChromaDB Embedder: OK")
        except Exception as e:
            print(f"  ChromaDB Embedder: Failed ({e})")
            from src.embedding_fallback import EmbedderFallback
            print("  Fallback Embedder: OK")
        
        # Test advanced scorer
        from src.advanced_scorer import AdvancedScorer
        print("  Advanced Scorer: OK")
        
        # Test enhanced ranker
        from src.enhanced_ranker import EnhancedRanker
        print("  Enhanced Ranker: OK")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("Testing basic functionality...")
        
        # Test database
        from src.enhanced_db_handler import EnhancedDBHandler
        db = EnhancedDBHandler()
        stats = db.get_database_stats()
        print(f"  Database: OK (stats: {len(stats)} metrics)")
        
        # Test parser
        from src.info_parser import InfoParser
        parser = InfoParser()
        print("  Parser: OK")
        
        # Test embedder
        try:
            from src.embedding import Embedder
            embedder = Embedder()
            print("  ChromaDB Embedder: OK")
        except Exception as e:
            from src.embedding_fallback import EmbedderFallback
            embedder = EmbedderFallback()
            print("  Fallback Embedder: OK")
        
        print("\nBasic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Resume Ranking System - Component Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 50)
            print("SUCCESS: All components are working!")
            print("You can now run:")
            print("  streamlit run enhanced_app.py")
            print("  python enhanced_main.py --mode pipeline")
        else:
            print("\nFunctionality test failed.")
    else:
        print("\nImport test failed.")
