import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.persona_analyzer import PersonaProfile
from src.job_parser import JobParser
import os

class RelevanceScorer:
    """
    Scores document sections based on their relevance to a given persona and job,
    using a sentence-transformer model for semantic understanding.
    """

    def __init__(self, persona_profile: PersonaProfile, job_details: dict, model_name: str = 'all-MiniLM-L6-v2'):
        self.persona = persona_profile
        self.job = job_details
        # Cache the model to avoid reloading it multiple times
        model_path = os.path.join('models', model_name)
        if not os.path.exists(model_path):
            print(f"Downloading and caching model {model_name} to {model_path}")
            os.makedirs(model_path, exist_ok=True)
            self.model = SentenceTransformer(model_name)
            self.model.save(model_path)
        else:
            print(f"Loading cached model from {model_path}")
            self.model = SentenceTransformer(model_path)

    def _apply_domain_boosters(self, score: float, section_text: str) -> float:
        """Applies domain-specific boosts and penalties to the score."""
        boost = 0.0
        text_lower = section_text.lower()

        # General-purpose boosters
        if any(kw in text_lower for kw in ['guide', 'how-to', 'steps', 'checklist']):
            boost += 0.1
        if any(kw in text_lower for kw in ['top 10', 'best practices', 'recommendations']):
            boost += 0.05

        # Domain-specific logic
        if self.persona.domain == 'travel':
            if 'group' in self.job.get('keywords', []) and 'group' in text_lower:
                boost += 0.2
            if 'budget' in self.job.get('keywords', []) and any(kw in text_lower for kw in ['budget', 'affordable', 'cheap']):
                boost += 0.15
            if 'luxury' in text_lower or 'solo travel' in text_lower:
                boost -= 0.1 # Penalty for irrelevant content

        elif self.persona.domain == 'hr':
            if 'compliance' in text_lower or 'policy' in text_lower:
                boost += 0.2
            if 'form' in text_lower or 'template' in text_lower:
                boost += 0.15

        elif self.persona.domain == 'academic':
            if any(kw in text_lower for kw in ['methodology', 'dataset', 'benchmark']):
                boost += 0.2
            if 'review' in text_lower or 'survey' in text_lower:
                boost += 0.1

        # Apply the boost and clamp the score between 0 and 1
        return min(max(score + boost, 0.0), 1.0)

    def score_section(self, section_text: str) -> float:
        """
        Calculates a relevance score for a single document section.

        Args:
            section_text: The text content of the document section.

        Returns:
            A relevance score between 0 and 1.
        """
        # Combine persona and job keywords for a comprehensive query
        query_keywords = self.persona.keywords + self.job['keywords']
        query_text = ' '.join(query_keywords)

        # Encode the query and the section text
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        section_embedding = self.model.encode(section_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = util.pytorch_cos_sim(query_embedding, section_embedding).item()

        # Apply domain-specific boosters
        final_score = self._apply_domain_boosters(similarity_score, section_text)

        return final_score

    def score_sections(self, sections: list) -> list:
        """
        Scores a list of document sections using sentence transformers.

        Args:
            sections: A list of strings, where each string is a section's text.

        Returns:
            A list of scores corresponding to each section.
        """
        if not sections:
            return []

        query_keywords = self.persona.keywords + self.job['keywords']
        query_text = ' '.join(query_keywords)

        # Encode the query and all sections
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        section_embeddings = self.model.encode(sections, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_scores = util.pytorch_cos_sim(query_embedding, section_embeddings).flatten()

        # Apply domain boosters to each section's score
        final_scores = [self._apply_domain_boosters(score.item(), section) for score, section in zip(similarity_scores, sections)]

        return final_scores

# Example Usage
if __name__ == '__main__':
    # Mock Persona and Job data
    persona_data = {
        'role': 'Travel Planner',
        'domain': 'travel',
        'expertise_level': 'intermediate',
        'keywords': ['travel', 'planning', 'group', 'budget'],
        'focus_areas': ['group_travel', 'budget_planning'],
        'confidence_score': 0.8,
        'extracted_entities': [],
        'semantic_keywords': [],
        'tfidf_keywords': []
    }
    persona_profile = PersonaProfile(**persona_data)

    job_parser = JobParser()
    job_text = "Plan a trip of 4 days for a group of 10 college friends."
    job_details = job_parser.parse_job(job_text)

    # Initialize the scorer
    scorer = RelevanceScorer(persona_profile, job_details)

    # Mock document sections
    sections_to_score = [
        "This section is about the history of the South of France.",
        "A guide to budget-friendly group travel in Europe.",
        "Details on luxury solo travel packages.",
        "A comprehensive list of restaurants and hotels in Nice."
    ]

    # Score a single section
    score = scorer.score_section(sections_to_score[1])
    print(f"Score for a single relevant section: {score:.4f}")

    # Score multiple sections
    scores = scorer.score_sections(sections_to_score)
    print("\nScores for multiple sections:")
    for i, s in enumerate(scores):
        print(f"  Section {i+1}: {s:.4f}")
