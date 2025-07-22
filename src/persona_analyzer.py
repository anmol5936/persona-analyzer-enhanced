import re
import json
import ssl
import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import pandas as pd

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data if not present
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    datasets = [
        'punkt',
        'stopwords', 
        'averaged_perceptron_tagger',
        'maxent_ne_chunker', 
        'words', 
        'wordnet', 
        'omw-1.4'
    ]
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
        except LookupError:
            try:
                print(f"Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
            except Exception as e:
                print(f"Failed to download {dataset}: {e}")
                # Try to download the 'punkt_tab' as a fallback
                if dataset == 'punkt':
                    try:
                        nltk.download('punkt_tab', quiet=True)
                    except Exception as e2:
                        print(f"Failed to download punkt_tab as well: {e2}")
                continue

# Initialize NLTK data
ensure_nltk_data()

@dataclass
class PersonaProfile:
    """Structured persona profile output"""
    role: str
    domain: str
    expertise_level: str
    keywords: List[str]
    focus_areas: List[str]
    confidence_score: float
    extracted_entities: List[str] = None
    semantic_keywords: List[str] = None
    tfidf_keywords: List[str] = None

class PersonaAnalyzer:
    """
    Enhanced Persona Analyzer with better error handling and improved accuracy
    """
    
    def __init__(self):
        # Initialize NLTK components with better error handling
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Stopwords not found, downloading...")
            nltk.download('stopwords', quiet=False)
            self.stop_words = set(stopwords.words('english'))
        
        # Enhanced stop words - be more selective
        self.stop_words.update(['role', 'position', 'job', 'work', 'years', 'experience'])
        
        # Enhanced role keywords with better categorization
        self.role_keywords = {
            'travel': ['travel', 'planner', 'agent', 'guide', 'coordinator', 'advisor', 
                      'consultant', 'specialist', 'manager', 'tourism', 'destination'],
            'hr': ['human', 'resources', 'hr', 'recruiter', 'talent', 'employee', 
                   'personnel', 'staffing', 'recruitment', 'hiring', 'acquisition'],
            'academic': ['teacher', 'professor', 'educator', 'instructor', 'researcher', 
                        'student', 'academic', 'faculty', 'phd', 'research', 'university', 
                        'scholar', 'literature', 'computational', 'biology'],
            'technology': ['developer', 'engineer', 'programmer', 'analyst', 'architect', 
                          'specialist', 'technician', 'data', 'scientist', 'machine', 
                          'learning', 'artificial', 'intelligence'],
            'healthcare': ['doctor', 'nurse', 'medical', 'healthcare', 'physician', 
                          'practitioner', 'clinician', 'health', 'clinical', 'biology', 
                          'biomedical', 'drug', 'discovery'],
            'business': ['manager', 'executive', 'analyst', 'consultant', 'director', 
                        'administrator', 'coordinator', 'business'],
            'finance': ['accountant', 'financial', 'banker', 'analyst', 'advisor', 
                       'controller', 'treasurer', 'finance'],
            'marketing': ['marketer', 'specialist', 'manager', 'coordinator', 
                         'strategist', 'analyst', 'executive', 'marketing']
        }
        
        self.expertise_indicators = {
            'beginner': ['new', 'junior', 'entry', 'trainee', 'novice', 'learning', 
                        'starting', 'fresh', 'beginning', '1', '2'],
            'intermediate': ['experienced', 'skilled', 'competent', 'proficient', 
                           'capable', 'seasoned', 'practiced', '3', '4', '5'],
            'expert': ['senior', 'expert', 'master', 'specialist', 'lead', 'chief', 
                      'advanced', 'veteran', 'principal', '6', '7', '8', '9', '10', 
                      'over', 'comprehensive', 'phd']
        }
        
        # Enhanced domain focus areas
        self.domain_focus_areas = {
            'travel': {
                'group_travel': ['group', 'team', 'family', 'corporate', 'tour'],
                'budget_planning': ['budget', 'cost', 'affordable', 'financial', 'management'],
                'destinations': ['destination', 'location', 'international', 'research'],
                'logistics': ['coordination', 'planning', 'itinerary', 'booking']
            },
            'hr': {
                'recruitment': ['hiring', 'recruitment', 'talent', 'acquisition'],
                'compliance': ['policy', 'compliance', 'management', 'governance'],
                'employee_relations': ['employee', 'relations', 'performance'],
                'training': ['training', 'development', 'programs']
            },
            'academic': {
                'research': ['research', 'literature', 'review', 'methodologies'],
                'computational': ['computational', 'machine', 'learning', 'neural'],
                'biology': ['biology', 'bioinformatics', 'drug', 'discovery'],
                'publications': ['papers', 'publications', 'benchmarks']
            },
            'technology': {
                'machine_learning': ['machine', 'learning', 'artificial', 'intelligence'],
                'data_science': ['data', 'science', 'analytics', 'modeling'],
                'software_development': ['development', 'programming', 'engineering'],
                'research': ['research', 'algorithms', 'neural', 'networks']
            }
        }
    
    def extract_basic_keywords(self, text: str) -> List[str]:
        """Extract basic keywords with improved error handling"""
        try:
            # Clean and tokenize
            text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
            tokens = word_tokenize(text_clean)
            
            # Get POS tags with error handling
            try:
                pos_tags = pos_tag(tokens)
            except LookupError as e:
                print(f"POS tagging error: {e}")
                # Fallback: treat all words as nouns
                pos_tags = [(token, 'NN') for token in tokens]
            
            # Filter for meaningful words
            relevant_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBG']
            keywords = []
            
            for word, pos in pos_tags:
                if (len(word) > 2 and 
                    word.isalpha() and
                    word not in self.stop_words):
                    
                    # Lemmatize with error handling
                    try:
                        lemmatized = self.lemmatizer.lemmatize(word)
                    except:
                        lemmatized = word
                    
                    if lemmatized not in keywords:
                        keywords.append(lemmatized)
            
            # Count frequency and return top keywords
            word_freq = Counter(keywords)
            return [word for word, count in word_freq.most_common(15)]
            
        except Exception as e:
            print(f"Error in basic keyword extraction: {e}")
            # Robust fallback
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            filtered_words = [w for w in words if w not in self.stop_words]
            return list(set(filtered_words))[:10]
    
    def extract_entities_enhanced(self, text: str) -> List[str]:
        """Enhanced entity extraction with better patterns"""
        entities = []
        
        # Technical term patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',  # Multi-word capitalized terms
            r'\b(Graph Neural Networks?)\b',
            r'\b(Machine Learning)\b',
            r'\b(Artificial Intelligence)\b',
            r'\b(Deep Learning)\b',
            r'\b(Natural Language Processing)\b',
            r'\b(Computer Vision)\b',
            r'\b(Data Science)\b',
            r'\b(Computational Biology)\b',
            r'\b(Drug Discovery)\b',
            r'\b(Human Resources)\b',
            r'\b(PhD Researcher)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and filter short entities
        unique_entities = []
        for entity in entities:
            if len(entity) > 3 and entity not in unique_entities:
                unique_entities.append(entity)
        
        return unique_entities[:10]
    
    def extract_keywords_tfidf_improved(self, text: str) -> List[str]:
        """Improved TF-IDF extraction with better reference corpus"""
        try:
            # Create domain-specific reference corpus
            reference_texts = [
                "research analysis methodology data science machine learning artificial intelligence",
                "travel planning coordination booking destinations logistics budget management",
                "human resources recruitment talent management employee relations training development",
                "computational biology bioinformatics drug discovery neural networks graph algorithms",
                "senior expert specialist advanced professional experienced skilled competent",
                "project management leadership strategic planning business analysis consulting"
            ]
            
            # Combine with input text
            corpus = [text] + reference_texts
            
            # Configure TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams
                min_df=1,
                max_df=0.8,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for input text
            scores = tfidf_matrix[0].toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter and return meaningful keywords
            meaningful_keywords = []
            for kw, score in keyword_scores:
                if score > 0 and len(kw) > 2:
                    meaningful_keywords.append(kw)
                if len(meaningful_keywords) >= 15:
                    break
            
            return meaningful_keywords
            
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []
    
    def extract_keywords_comprehensive(self, text: str) -> Dict[str, List[str]]:
        """Comprehensive keyword extraction with multiple methods"""
        # Get keywords from different methods
        basic_keywords = self.extract_basic_keywords(text)
        tfidf_keywords = self.extract_keywords_tfidf_improved(text)
        entities = self.extract_entities_enhanced(text)
        
        # Combine with intelligent weighting
        keyword_scores = Counter()
        
        # Weight basic keywords
        for kw in basic_keywords[:10]:
            keyword_scores[kw] += 2
        
        # Weight TF-IDF keywords more heavily
        for kw in tfidf_keywords[:10]:
            keyword_scores[kw] += 3
        
        # Add entity bonus
        for entity in entities:
            for word in entity.lower().split():
                if len(word) > 2:
                    keyword_scores[word] += 1
        
        # Get final combined keywords
        combined_keywords = [kw for kw, score in keyword_scores.most_common(12)]
        
        return {
            'combined': combined_keywords,
            'basic': basic_keywords,
            'tfidf': tfidf_keywords,
            'entities': entities,
            'semantic': []  # Placeholder for future enhancement
        }
    
    def identify_role_enhanced(self, text: str) -> str:
        """Enhanced role identification with better patterns"""
        text_lower = text.lower()
        
        # Comprehensive role patterns
        patterns = [
            r'role[:\s]+([^,\n\.!?]+)',
            r'position[:\s]+([^,\n\.!?]+)',
            r'job title[:\s]+([^,\n\.!?]+)',
            r'i am a[n]?\s+([^,\n\.!?]+)',
            r'work as a[n]?\s+([^,\n\.!?]+)',
            r'(senior|lead|principal|chief)\s+([^,\n\.!?]+)',
            r'(phd|dr\.?)\s+(researcher|student)\s+in\s+([^,\n\.!?]+)',
            r'([a-z\s]*(?:researcher|scientist|manager|coordinator|specialist|analyst|planner|developer|engineer))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) > 1:
                    # Handle multiple groups
                    role_parts = [g for g in match.groups() if g]
                    role = ' '.join(role_parts)
                else:
                    role = match.group(1)
                
                # Clean and format role
                role = re.sub(r'\s+', ' ', role.strip())
                role = re.sub(r'\b(in|with|for|and)\b.*', '', role)  # Remove trailing prepositions
                
                # Capitalize properly
                words = role.split()
                formatted_words = []
                for word in words:
                    if word.lower() in ['phd', 'hr', 'ai', 'ml']:
                        formatted_words.append(word.upper())
                    elif word.lower() in ['in', 'of', 'with', 'for', 'and']:
                        formatted_words.append(word.title())
                    else:
                        formatted_words.append(word.title())
                
                formatted_role = ' '.join(formatted_words)
                return formatted_role if len(formatted_role) < 100 else formatted_role[:100]
        
        return "Professional"
    
    def detect_domain_enhanced(self, text: str) -> str:
        """Enhanced domain detection with better scoring"""
        text_lower = text.lower()
        domain_scores = {}
        
        # Score domains based on keyword presence
        for domain, keywords in self.role_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact matches get higher scores
                exact_matches = len(re.findall(rf'\b{keyword}\b', text_lower))
                partial_matches = text_lower.count(keyword) - exact_matches
                
                score += exact_matches * 3 + partial_matches * 1
            
            domain_scores[domain] = score
        
        # Special domain combinations
        if any(term in text_lower for term in ['computational biology', 'bioinformatics', 'drug discovery']):
            domain_scores['academic'] = domain_scores.get('academic', 0) + 8
        
        if any(term in text_lower for term in ['data scientist', 'machine learning', 'artificial intelligence']):
            domain_scores['technology'] = domain_scores.get('technology', 0) + 8
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    def classify_expertise_enhanced(self, text: str) -> str:
        """Enhanced expertise classification"""
        text_lower = text.lower()
        
        # Experience patterns with better parsing
        experience_patterns = [
            (r'(\d+)\+?\s*years?\s*of\s*(experience|expertise)', 
             lambda x: 'expert' if int(x) >= 6 else 'intermediate' if int(x) >= 3 else 'beginner'),
            (r'over\s*(\d+)\s*years?', 
             lambda x: 'expert' if int(x) >= 5 else 'intermediate'),
            (r'more than\s*(\d+)\s*years?', 
             lambda x: 'expert' if int(x) >= 5 else 'intermediate'),
            (r'(\d+)\s*years?\s*of\s*(research|work|professional)', 
             lambda x: 'expert' if int(x) >= 5 else 'intermediate' if int(x) >= 3 else 'beginner')
        ]
        
        for pattern, classifier in experience_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    years = int(match.group(1))
                    return classifier(years)
                except (ValueError, IndexError):
                    continue
        
        # Keyword-based scoring
        level_scores = {'beginner': 0, 'intermediate': 0, 'expert': 0}
        
        for level, indicators in self.expertise_indicators.items():
            for indicator in indicators:
                count = text_lower.count(indicator)
                level_scores[level] += count
        
        # Additional expert indicators
        expert_terms = ['phd', 'doctorate', 'comprehensive', 'advanced', 'senior', 'lead', 'principal', 'expert']
        for term in expert_terms:
            if term in text_lower:
                level_scores['expert'] += 2
        
        max_score = max(level_scores.values())
        if max_score == 0:
            return 'intermediate'
        
        return max(level_scores, key=level_scores.get)
    
    def extract_focus_areas_enhanced(self, text: str, domain: str) -> List[str]:
        """Enhanced focus area extraction"""
        text_lower = text.lower()
        focus_areas = []
        
        if domain in self.domain_focus_areas:
            area_scores = {}
            
            for area, keywords in self.domain_focus_areas[domain].items():
                score = 0
                for keyword in keywords:
                    # Exact word matches
                    exact_matches = len(re.findall(rf'\b{keyword}\b', text_lower))
                    score += exact_matches * 2
                    
                    # Partial matches
                    partial_matches = text_lower.count(keyword) - exact_matches
                    score += partial_matches
                
                if score > 0:
                    area_scores[area] = score
            
            # Sort and return top areas
            sorted_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)
            focus_areas = [area for area, score in sorted_areas[:4]]
        
        return focus_areas
    
    def calculate_confidence_enhanced(self, role: str, domain: str, expertise_level: str, 
                                    keywords: Dict[str, List[str]], text_length: int) -> float:
        """Enhanced confidence calculation"""
        score = 0.0
        
        # Role identification (max 0.3)
        if role != "Professional":
            score += 0.2
            if len(role.split()) >= 2:  # Multi-word role
                score += 0.1
        
        # Domain confidence (max 0.2)
        if domain != 'general':
            score += 0.2
        
        # Keywords quality (max 0.3)
        combined_keywords = keywords.get('combined', [])
        if len(combined_keywords) >= 10:
            score += 0.3
        elif len(combined_keywords) >= 6:
            score += 0.2
        elif len(combined_keywords) >= 3:
            score += 0.1
        
        # Text richness (max 0.1)
        if text_length > 150:
            score += 0.1
        elif text_length > 75:
            score += 0.05
        
        # Entity bonus (max 0.05)
        if keywords.get('entities'):
            score += 0.05
        
        # TF-IDF bonus (max 0.05)
        if keywords.get('tfidf'):
            score += 0.05
        
        return min(max(score, 0.4), 1.0)  # Range: 0.4 to 1.0
    
    def analyze_persona(self, persona_text: str) -> PersonaProfile:
        """Main analysis method with comprehensive error handling"""
        try:
            # Extract keywords
            keyword_results = self.extract_keywords_comprehensive(persona_text)
            
            # Extract profile components
            role = self.identify_role_enhanced(persona_text)
            domain = self.detect_domain_enhanced(persona_text)
            expertise_level = self.classify_expertise_enhanced(persona_text)
            focus_areas = self.extract_focus_areas_enhanced(persona_text, domain)
            
            # Calculate confidence
            confidence_score = self.calculate_confidence_enhanced(
                role, domain, expertise_level, keyword_results, len(persona_text)
            )
            
            return PersonaProfile(
                role=role,
                domain=domain,
                expertise_level=expertise_level,
                keywords=keyword_results['combined'],
                focus_areas=focus_areas,
                confidence_score=confidence_score,
                extracted_entities=keyword_results['entities'],
                semantic_keywords=keyword_results.get('semantic', []),
                tfidf_keywords=keyword_results['tfidf']
            )
            
        except Exception as e:
            print(f"Critical error in persona analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback analysis
            return PersonaProfile(
                role="Unknown Professional",
                domain="general",
                expertise_level="intermediate",
                keywords=[],
                focus_areas=[],
                confidence_score=0.3,
                extracted_entities=[],
                semantic_keywords=[],
                tfidf_keywords=[]
            )
    
    def analyze_persona_dict(self, persona_dict: Dict) -> PersonaProfile:
        """Analyze persona from dictionary with better text extraction"""
        text_parts = []
        
        # Extract text in order of importance
        if 'role' in persona_dict:
            text_parts.append(persona_dict['role'])
        
        if 'description' in persona_dict:
            text_parts.append(persona_dict['description'])
        
        if 'expertise' in persona_dict:
            text_parts.append(f"Expertise in {persona_dict['expertise']}")
        
        if 'experience' in persona_dict:
            text_parts.append(persona_dict['experience'])
        
        if 'skills' in persona_dict:
            if isinstance(persona_dict['skills'], list):
                text_parts.append(f"Skilled in {', '.join(persona_dict['skills'])}")
            else:
                text_parts.append(f"Skilled in {persona_dict['skills']}")
        
        # Join with proper spacing
        persona_text = ". ".join(text_parts)
        return self.analyze_persona(persona_text)

# Test the fixed analyzer
if __name__ == "__main__":
    analyzer = PersonaAnalyzer()
    
    test_text = "I am a senior data scientist with 8 years of experience in machine learning and artificial intelligence, specializing in healthcare applications and natural language processing."
    
    try:
        profile = analyzer.analyze_persona(test_text)
        print("Enhanced Analyzer Test Results:")
        print("="*60)
        
        result_dict = asdict(profile)
        for key, value in result_dict.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    except Exception as e:
        print(f"Test failed: {e}")