"""
Enhanced Resume Ranking Application with Advanced Features
"""

import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdfplumber
import numpy as np
from datetime import datetime
import json

from src.enhanced_db_handler import EnhancedDBHandler
from src.enhanced_ranker import EnhancedRanker
from src.info_parser import InfoParser

# Try to import ChromaDB embedder, fallback to in-memory if it fails
try:
    from src.embedding import Embedder
    EMBEDDER_TYPE = "chromadb"
except Exception as e:
    print(f"ChromaDB embedder failed to load: {e}")
    print("Using fallback in-memory embedder")
    from src.embedding_fallback import EmbedderFallback as Embedder
    EMBEDDER_TYPE = "fallback"

# Page configuration
st.set_page_config(
    page_title="Advanced Resume Ranking System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-breakdown {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.db = EnhancedDBHandler()
    st.session_state.ranker = EnhancedRanker()
    st.session_state.parser = InfoParser()
    st.session_state.embedder = Embedder()

def load_file_content(file):
    """Load content from uploaded file"""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return file.read().decode("utf-8")

def display_analytics_dashboard(session_id):
    """Display comprehensive analytics dashboard"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get analytics data
    analytics = st.session_state.ranker.get_ranking_insights(session_id)
    session_analytics = st.session_state.db.get_session_analytics(session_id)
    
    if not analytics:
        st.warning("No analytics data available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Candidates",
            session_analytics.get('total_resumes', 0),
            delta=None
        )
    
    with col2:
        avg_score = session_analytics.get('score_statistics', {}).get('average_score', 0)
        st.metric(
            "Average Score",
            f"{avg_score:.3f}",
            delta=f"{avg_score:.1%}"
        )
    
    with col3:
        excellent_candidates = len([r for r in analytics.get('top_candidates', []) if r['final_score'] >= 0.8])
        st.metric(
            "Excellent Matches",
            excellent_candidates,
            delta=f"{excellent_candidates/session_analytics.get('total_resumes', 1)*100:.1f}%"
        )
    
    with col4:
        top_score = max([r['final_score'] for r in analytics.get('top_candidates', [])], default=0)
        st.metric(
            "Best Match Score",
            f"{top_score:.3f}",
            delta=f"{top_score:.1%}"
        )
    
    # Score distribution chart
    st.markdown("### 📈 Score Distribution")
    
    if analytics.get('top_candidates'):
        scores = [r['final_score'] for r in analytics['top_candidates']]
        names = [r['filename'] for r in analytics['top_candidates']]
        
        fig = px.bar(
            x=names,
            y=scores,
            title="Candidate Scores",
            color=scores,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            xaxis_title="Candidates",
            yaxis_title="Final Score",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Score breakdown radar chart
    if analytics.get('top_candidates'):
        st.markdown("### 🎯 Score Breakdown Analysis")
        
        top_candidate = analytics['top_candidates'][0]
        
        categories = ['Semantic', 'Skills', 'Experience', 'Education', 'Structure', 'TF-IDF', 'Keywords']
        values = [
            top_candidate['semantic_score'],
            top_candidate['skill_score'],
            top_candidate['experience_score'],
            top_candidate['education_score'],
            top_candidate['structure_score'],
            top_candidate['tfidf_score'],
            top_candidate['keyword_score']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=top_candidate['filename'],
            line_color='rgb(102, 126, 234)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Top Candidate Score Breakdown"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_ranking_results(session_id):
    """Display enhanced ranking results"""
    st.markdown("## 🏆 Enhanced Ranking Results")
    
    # Get ranked resumes
    results = st.session_state.db.get_ranked_resumes_enhanced(session_id, top_k=20)
    
    if not results:
        st.warning("No ranking results found")
        return
    
    # Display top candidates with detailed breakdown
    def _to_float(x):
        if isinstance(x, (bytes, bytearray)):
            try:
                x = x.decode('utf-8', errors='ignore')
            except Exception:
                return 0.0
        try:
            return float(x)
        except Exception:
            return 0.0

    for i, result in enumerate(results[:10], 1):
        score_value = _to_float(result.get('final_score', 0.0))
        title_name = result.get('name') or result['filename']
        title_email = result.get('email') or "-"
        title_phone = result.get('phone') or "-"
        title_exp = _to_float(result.get('experience', 0.0))
        with st.expander(f"#{i} {title_name} | {title_email} | {title_phone} | Exp: {title_exp:.1f} yrs - Score: {score_value:.3f}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Name:** {result['name']}")
                st.markdown(f"**Email:** {result['email']}")
                st.markdown(f"**Phone:** {result['phone']}")
                st.markdown(f"**Experience:** {result['experience']} years")
                st.markdown(f"**Education:** {result['education_level']}")
            
            with col2:
                # Score breakdown
                st.markdown("**Score Breakdown:**")
                
                scores_data = {
                    'Metric': ['Semantic', 'Skills', 'Experience', 'Education', 'Structure', 'TF-IDF', 'Keywords'],
                    'Score': [
                        _to_float(result['semantic_score']),
                        _to_float(result['skill_score']),
                        _to_float(result['experience_score']),
                        _to_float(result['education_score']),
                        _to_float(result['structure_score']),
                        _to_float(result['tfidf_score']),
                        _to_float(result['keyword_score'])
                    ]
                }
                
                df_scores = pd.DataFrame(scores_data)
                
                fig = px.bar(
                    df_scores,
                    x='Metric',
                    y='Score',
                    title=f"Score Breakdown - {result['filename']}",
                    color='Score',
                    color_continuous_scale="RdYlBu"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Skills extracted
            if result['skills_extracted']:
                st.markdown("**Skills Extracted:**")
                for category, skills in result['skills_extracted'].items():
                    if skills:
                        st.markdown(f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}")

def display_insights_and_recommendations(session_id):
    """Display insights and recommendations"""
    st.markdown("## 💡 Insights & Recommendations")
    
    insights = st.session_state.ranker.get_ranking_insights(session_id)
    
    if not insights:
        st.warning("No insights available")
        return
    
    # Score gaps
    if insights.get('score_gaps'):
        st.markdown("### ⚠️ Significant Score Gaps")
        for gap in insights['score_gaps']:
            st.info(f"**Position {gap['position']}:** {gap['candidate']} has a significant lead over {gap['next_candidate']} (gap: {gap['score_gap']:.3f})")
    
    # Recommendations
    if insights.get('recommendations'):
        st.markdown("### 📋 Recommendations")
        for recommendation in insights['recommendations']:
            st.success(f"💡 {recommendation}")

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced Resume Ranking System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Database stats
    st.sidebar.markdown("### 📊 Database Statistics")
    stats = st.session_state.db.get_database_stats()
    st.sidebar.metric("Total Resumes", stats.get('resumes_count', 0))
    st.sidebar.metric("Total Jobs", stats.get('jobs_count', 0))
    st.sidebar.metric("Total Sessions", stats.get('sessions_count', 0))
    
    # Configuration
    st.sidebar.markdown("### ⚙️ Scoring Configuration")
    
    with st.sidebar.expander("Adjust Scoring Weights"):
        semantic_weight = st.slider("Semantic Similarity", 0.0, 0.5, 0.25, 0.05)
        skill_weight = st.slider("Skill Matching", 0.0, 0.5, 0.25, 0.05)
        experience_weight = st.slider("Experience", 0.0, 0.3, 0.15, 0.05)
        education_weight = st.slider("Education", 0.0, 0.2, 0.10, 0.05)
        keyword_weight = st.slider("Keywords", 0.0, 0.3, 0.15, 0.05)
        structure_weight = st.slider("Structure", 0.0, 0.1, 0.05, 0.05)
        tfidf_weight = st.slider("TF-IDF", 0.0, 0.1, 0.05, 0.05)
        
        # Normalize weights
        total = semantic_weight + skill_weight + experience_weight + education_weight + keyword_weight + structure_weight + tfidf_weight
        if total > 0:
            semantic_weight /= total
            skill_weight /= total
            experience_weight /= total
            education_weight /= total
            keyword_weight /= total
            structure_weight /= total
            tfidf_weight /= total
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "🏆 Results", "📊 Analytics", "🔧 Advanced"])
    
    with tab1:
        st.markdown("## 📤 Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_file = st.file_uploader(
                "Upload Job Description",
                type=["txt", "pdf"],
                help="Upload the job description file"
            )
        
        with col2:
            resume_files = st.file_uploader(
                "Upload Resumes",
                type=["txt", "pdf"],
                accept_multiple_files=True,
                help="Upload multiple resume files"
            )
        
        # Process button
        if st.button("🚀 Run Advanced Ranking", type="primary"):
            if not job_file or not resume_files:
                st.error("⚠️ Please upload both a job description and at least one resume.")
            else:
                with st.spinner("🔄 Processing files and running advanced ranking..."):
                    try:
                        # Process job description
                        job_text = load_file_content(job_file)
                        # Use parser only to extract info; do NOT rely on legacy DB IDs
                        parsed_job = st.session_state.parser.parse_job(job_text, job_file.name)

                        # Insert job into enhanced DB and create session
                        enhanced_job_id = st.session_state.db.insert_job_enhanced(
                            parsed_job["job_name"],
                            job_text,
                            {"exp_required": parsed_job.get("exp_required", 0.0)}
                        )
                        session_id = st.session_state.db.insert_session_enhanced(
                            enhanced_job_id,
                            f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )

                        # Process job embedding using enhanced job id
                        st.session_state.embedder.process_job(
                            enhanced_job_id,
                            job_text,
                            {"job_id": enhanced_job_id, "job_name": parsed_job["job_name"], "exp_required": parsed_job.get("exp_required", 0.0), "text": job_text}
                        )
                        
                        # Process resumes
                        resumes_data = []
                        progress_bar = st.progress(0)
                        
                        for i, file in enumerate(resume_files):
                            resume_text = load_file_content(file)
                            # Parse to extract fields only (legacy insert ignored)
                            parsed_res = st.session_state.parser.parse_resume(resume_text, file.name)

                            # Insert resume into enhanced DB
                            enhanced_resume_id = st.session_state.db.insert_resume_enhanced(
                                parsed_res["filename"],
                                resume_text,
                                {
                                    "name": parsed_res.get("name"),
                                    "email": parsed_res.get("email"),
                                    "phone": parsed_res.get("phone"),
                                    "experience": parsed_res.get("experience", 0.0),
                                    "education_level": st.session_state.ranker.scorer.extract_education(resume_text),
                                    "skills": st.session_state.ranker.scorer.extract_skills(resume_text)
                                }
                            )

                            # Process resume embedding with enhanced id (ensure enhanced id wins)
                            resume_metadata = {**parsed_res, "resume_id": enhanced_resume_id, "text": resume_text}
                            st.session_state.embedder.process_resume(
                                enhanced_resume_id,
                                resume_text,
                                resume_metadata
                            )

                            # Ensure enhanced resume_id is used for ranking data
                            resumes_data.append({**parsed_res, "resume_id": enhanced_resume_id, "text": resume_text})
                            progress_bar.progress((i + 1) / len(resume_files))
                        
                        # Run enhanced ranking
                        ranking_result = st.session_state.ranker.rank_session_enhanced(
                            session_id, resumes_data
                        )
                        
                        st.success(f"✅ Successfully ranked {len(resumes_data)} resumes!")
                        st.session_state.current_session_id = session_id
                        st.session_state.ranking_completed = True
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
    
    with tab2:
        if 'current_session_id' in st.session_state and st.session_state.get('ranking_completed', False):
            display_ranking_results(st.session_state.current_session_id)
        else:
            st.info("👆 Please upload files and run ranking first.")
    
    with tab3:
        if 'current_session_id' in st.session_state and st.session_state.get('ranking_completed', False):
            display_analytics_dashboard(st.session_state.current_session_id)
        else:
            st.info("👆 Please upload files and run ranking first.")
    
    with tab4:
        st.markdown("## 🔧 Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗄️ Database Management")
            
            if st.button("📊 View Database Statistics"):
                stats = st.session_state.db.get_database_stats()
                st.json(stats)
            
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.db.clear_all_data()
                    st.success("✅ All data cleared!")
                    st.session_state.confirm_clear = False
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm clearing all data")
            
            if st.button("🔍 Search Similar Resumes"):
                query = st.text_input("Enter search query:")
                if query:
                    # This would implement semantic search
                    st.info("🔍 Semantic search feature coming soon!")
        
        with col2:
            st.markdown("### 📈 Performance Metrics")
            
            # Display performance metrics
            st.markdown("**Model Performance:**")
            st.metric("Embedding Model", "all-mpnet-base-v2")
            st.metric("Vector Dimension", "768")
            st.metric("ChromaDB Collections", "3")
            
            st.markdown("**Recent Activity:**")
            st.info("📊 Last ranking session completed successfully")
            st.info("🎯 Advanced scoring algorithms active")

if __name__ == "__main__":
    main()
