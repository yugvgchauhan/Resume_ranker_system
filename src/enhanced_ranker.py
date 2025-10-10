"""
Enhanced Ranking System with Advanced Algorithms
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from src.advanced_scorer import AdvancedScorer
from src.enhanced_db_handler import EnhancedDBHandler
import json

# Try to import ChromaDB embedder, fallback if it fails
try:
    from src.embedding import Embedder
    EMBEDDER_TYPE = "chromadb"
except Exception as e:
    print(f"ChromaDB embedder failed to load: {e}")
    print("Using fallback in-memory embedder")
    from src.embedding_fallback import EmbedderFallback as Embedder
    EMBEDDER_TYPE = "fallback"


class EnhancedRanker:
    """
    Enhanced ranking system with sophisticated algorithms and analytics
    """
    
    def __init__(self, config_path: str = None):
        self.db = EnhancedDBHandler()
        self.scorer = AdvancedScorer()
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration for ranking weights"""
        default_config = {
            'scoring_weights': {
                'semantic_weight': 0.25,
                'skill_match_weight': 0.25,
                'experience_weight': 0.15,
                'education_weight': 0.10,
                'keyword_weight': 0.15,
                'structure_weight': 0.05,
                'tfidf_weight': 0.05
            },
            'experience_penalty': {
                'over_qualified_threshold': 2.0,  # 2x required experience
                'over_qualified_penalty': 0.2
            },
            'education_boost': {
                'phd_boost': 0.1,
                'master_boost': 0.05,
                'bachelor_boost': 0.0
            },
            'skill_importance': {
                'programming_languages': 1.0,
                'frameworks': 0.8,
                'databases': 0.7,
                'cloud_platforms': 0.9,
                'tools': 0.6
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def rank_session_enhanced(self, session_id: int, resumes_data: List[Dict]) -> Dict:
        """
        Enhanced ranking with detailed analytics and insights
        """
        print(f"Starting enhanced ranking for session {session_id}")
        
        # Get job information
        job_info = self.db.get_job_by_session(session_id)
        if not job_info:
            print(f"No job found for session {session_id}")
            return {}
        
        job_id = job_info['job_id']
        
        # Get job text (ChromaDB or fallback)
        if EMBEDDER_TYPE == "chromadb":
            job_data = self.scorer.embedder.job_collection.get(
                ids=[str(job_id)], 
                include=["documents", "metadatas"]
            )
            
            if not job_data or not job_data.get('documents'):
                print(f"No job text found for job_id={job_id}")
                return {}
            
            job_text = job_data['documents'][0]
        else:
            # Fallback mode
            job_doc = self.scorer.embedder.get_job_document(job_id)
            if not job_doc:
                print(f"No job text found for job_id={job_id}")
                return {}
            job_text = job_doc['text']
        
        print(f"Job: {job_info['job_name']}")
        print(f"Processing {len(resumes_data)} resumes...")
        
        # Calculate scores for all resumes
        ranking_results = []
        detailed_scores = []
        
        for i, resume_data in enumerate(resumes_data):
            resume_id = resume_data['resume_id']
            resume_text = resume_data['text']
            
            print(f"  Processing resume {i+1}/{len(resumes_data)}: {resume_data.get('filename', 'Unknown')}")
            
            # Calculate comprehensive score
            score_result = self.scorer.calculate_comprehensive_score(
                job_text, resume_text, resume_data
            )
            
            # Apply additional enhancements
            enhanced_score = self._apply_enhancements(
                score_result, job_text, resume_text, resume_data
            )
            
            ranking_results.append({
                'resume_id': resume_id,
                'job_id': job_id,
                'final_score': enhanced_score['final_score'],
                'semantic_score': enhanced_score['breakdown']['semantic_similarity'],
                'skill_score': enhanced_score['breakdown']['skill_match'],
                'experience_score': enhanced_score['breakdown']['experience_match'],
                'education_score': enhanced_score['breakdown']['education_match'],
                'structure_score': enhanced_score['breakdown']['structure_score'],
                'tfidf_score': enhanced_score['breakdown']['tfidf_similarity'],
                'keyword_score': enhanced_score['breakdown']['keyword_overlap'],
                'rank_position': 0  # Will be set after sorting
            })
            
            detailed_scores.append({
                'resume_id': resume_id,
                'filename': resume_data.get('filename', 'Unknown'),
                'scores': enhanced_score
            })
        
        # Sort by final score
        ranking_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Set rank positions
        for i, result in enumerate(ranking_results):
            result['rank_position'] = i + 1
        
        # Store results in database
        self.db.store_rankings_enhanced(session_id, ranking_results)
        
        # Generate analytics
        analytics = self._generate_analytics(session_id, ranking_results, detailed_scores)
        
        print(f"Enhanced ranking completed for session {session_id}")
        
        return {
            'session_id': session_id,
            'total_resumes': len(ranking_results),
            'rankings': ranking_results,
            'analytics': analytics,
            'detailed_scores': detailed_scores
        }
    
    def _apply_enhancements(self, score_result: Dict, job_text: str, resume_text: str, resume_data: Dict) -> Dict:
        """Apply additional enhancements to the base score"""
        enhanced = score_result.copy()
        
        # Experience penalty for over-qualification
        exp_config = self.config['experience_penalty']
        job_exp = score_result['extracted_info']['job_experience']['total_years']
        resume_exp = score_result['extracted_info']['resume_experience']['total_years']
        
        if resume_exp > job_exp * exp_config['over_qualified_threshold']:
            penalty = exp_config['over_qualified_penalty']
            enhanced['final_score'] *= (1 - penalty)
            enhanced['breakdown']['experience_match'] *= (1 - penalty)
        
        # Education boost
        education_config = self.config['education_boost']
        resume_education = score_result['extracted_info']['resume_education']
        
        if resume_education == 'phd':
            boost = education_config['phd_boost']
        elif resume_education == 'master':
            boost = education_config['master_boost']
        elif resume_education == 'bachelor':
            boost = education_config['bachelor_boost']
        else:
            boost = 0.0
        
        if boost > 0:
            enhanced['final_score'] += boost
            enhanced['breakdown']['education_match'] += boost
        
        # Skill importance weighting
        skill_config = self.config['skill_importance']
        resume_skills = score_result['extracted_info']['resume_skills']
        job_skills = score_result['extracted_info']['job_skills']
        
        weighted_skill_score = 0.0
        total_weight = 0.0
        
        for category in resume_skills:
            if category in job_skills and category in skill_config:
                weight = skill_config[category]
                category_score = len(set(resume_skills[category]) & set(job_skills[category]))
                category_total = len(job_skills[category])
                
                if category_total > 0:
                    weighted_skill_score += (category_score / category_total) * weight
                    total_weight += weight
        
        if total_weight > 0:
            enhanced_skill_score = weighted_skill_score / total_weight
            enhanced['breakdown']['skill_match'] = enhanced_skill_score
            # Recalculate final score with enhanced skill score
            weights = self.scorer.config
            enhanced['final_score'] = (
                weights['semantic_weight'] * enhanced['breakdown']['semantic_similarity'] +
                weights['skill_match_weight'] * enhanced_skill_score +
                weights['experience_weight'] * enhanced['breakdown']['experience_match'] +
                weights['education_weight'] * enhanced['breakdown']['education_match'] +
                weights['structure_weight'] * enhanced['breakdown']['structure_score'] +
                weights['tfidf_weight'] * enhanced['breakdown']['tfidf_similarity'] +
                weights['keyword_weight'] * enhanced['breakdown']['keyword_overlap']
            )
        
        return enhanced
    
    def _generate_analytics(self, session_id: int, rankings: List[Dict], detailed_scores: List[Dict]) -> Dict:
        """Generate comprehensive analytics for the ranking session"""
        if not rankings:
            return {}
        
        scores = [r['final_score'] for r in rankings]
        
        analytics = {
            'score_statistics': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            },
            'score_distribution': {
                'excellent': len([s for s in scores if s >= 0.8]),
                'good': len([s for s in scores if 0.6 <= s < 0.8]),
                'average': len([s for s in scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in scores if s < 0.4])
            },
            'top_skills': self._analyze_top_skills(detailed_scores),
            'experience_analysis': self._analyze_experience(detailed_scores),
            'education_analysis': self._analyze_education(detailed_scores)
        }
        
        return analytics
    
    def _analyze_top_skills(self, detailed_scores: List[Dict]) -> Dict:
        """Analyze top skills across all resumes"""
        skill_counter = {}
        
        for score_data in detailed_scores:
            skills = score_data['scores']['extracted_info']['resume_skills']
            for category, skill_list in skills.items():
                for skill in skill_list:
                    if skill not in skill_counter:
                        skill_counter[skill] = 0
                    skill_counter[skill] += 1
        
        # Sort by frequency
        top_skills = sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'most_common': top_skills,
            'total_unique_skills': len(skill_counter)
        }
    
    def _analyze_experience(self, detailed_scores: List[Dict]) -> Dict:
        """Analyze experience distribution"""
        experiences = []
        
        for score_data in detailed_scores:
            exp = score_data['scores']['extracted_info']['resume_experience']['total_years']
            experiences.append(exp)
        
        if not experiences:
            return {}
        
        return {
            'mean_experience': np.mean(experiences),
            'median_experience': np.median(experiences),
            'experience_ranges': {
                '0-2 years': len([e for e in experiences if 0 <= e < 2]),
                '2-5 years': len([e for e in experiences if 2 <= e < 5]),
                '5-10 years': len([e for e in experiences if 5 <= e < 10]),
                '10+ years': len([e for e in experiences if e >= 10])
            }
        }
    
    def _analyze_education(self, detailed_scores: List[Dict]) -> Dict:
        """Analyze education distribution"""
        education_counter = {}
        
        for score_data in detailed_scores:
            education = score_data['scores']['extracted_info']['resume_education']
            education_counter[education] = education_counter.get(education, 0) + 1
        
        return education_counter
    
    def get_ranking_insights(self, session_id: int) -> Dict:
        """Get insights and recommendations for a ranking session"""
        rankings = self.db.get_ranked_resumes_enhanced(session_id, top_k=50)
        analytics = self.db.get_session_analytics(session_id)
        
        if not rankings:
            return {}
        
        insights = {
            'top_candidates': rankings[:5],
            'score_gaps': self._identify_score_gaps(rankings),
            'recommendations': self._generate_recommendations(rankings, analytics),
            'skill_gaps': self._identify_skill_gaps(rankings)
        }
        
        return insights
    
    def _identify_score_gaps(self, rankings: List[Dict]) -> List[Dict]:
        """Identify significant score gaps between candidates"""
        gaps = []
        
        def _to_float(x):
            try:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode('utf-8', errors='ignore')
                return float(x)
            except Exception:
                return 0.0

        for i in range(len(rankings) - 1):
            current_score = _to_float(rankings[i]['final_score'])
            next_score = _to_float(rankings[i + 1]['final_score'])
            gap = current_score - next_score
            
            if gap > 0.1:  # Significant gap threshold
                gaps.append({
                    'position': i + 1,
                    'candidate': rankings[i]['filename'],
                    'score_gap': gap,
                    'next_candidate': rankings[i + 1]['filename']
                })
        
        return gaps
    
    def _generate_recommendations(self, rankings: List[Dict], analytics: Dict) -> List[str]:
        """Generate recommendations based on ranking analysis"""
        recommendations = []
        
        # Check score distribution
        excellent_count = len([r for r in rankings if r['final_score'] >= 0.8])
        if excellent_count == 0:
            recommendations.append("Consider expanding the candidate pool - no excellent matches found")
        elif excellent_count < 3:
            recommendations.append("Limited excellent candidates - consider additional sourcing")
        
        # Check experience distribution
        if 'score_statistics' in analytics:
            avg_score = analytics['score_statistics'].get('average_score', 0)
            if avg_score < 0.5:
                recommendations.append("Overall candidate quality is low - review job requirements")
            elif avg_score > 0.7:
                recommendations.append("Strong candidate pool - consider narrowing criteria")
        
        return recommendations
    
    def _identify_skill_gaps(self, rankings: List[Dict]) -> Dict:
        """Identify common skill gaps in top candidates"""
        # This would require more detailed skill analysis
        # For now, return empty structure
        return {
            'missing_skills': [],
            'skill_frequency': {}
        }


# Test function
if __name__ == "__main__":
    ranker = EnhancedRanker()
    print("Enhanced Ranker initialized successfully!")
