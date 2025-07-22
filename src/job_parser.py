import re
from typing import Dict, List, Tuple

class JobParser:
    """
    Parses job-to-be-done descriptions to extract actionable requirements,
    constraints, and success criteria.
    """

    def __init__(self):
        # Define patterns for extracting key information
        self.action_verbs = [
            'plan', 'create', 'analyze', 'prepare', 'summarize', 'identify',
            'review', 'manage', 'develop', 'implement', 'extract'
        ]

        self.constraints_patterns = {
            'time': r'(\d+)\s*(days?|weeks?|months?)',
            'group_size': r'(group of|for)\s*(\d+)\s*(people|friends?|colleagues?|members?)',
            'budget': r'(budget|cost|price)\s*of\s*([\$€£]?\d+k?)',
            'demographics': r'(college friends|students|professionals|executives|seniors|beginners)'
        }

        self.deliverable_keywords = [
            'trip', 'menu', 'report', 'review', 'summary', 'analysis', 'plan',
            'document', 'form', 'presentation', 'itinerary'
        ]

    def parse_job(self, job_text: str) -> Dict:
        """
        Analyzes a job description to extract structured information.

        Args:
            job_text: The raw job description string.

        Returns:
            A dictionary containing extracted actions, deliverables, constraints,
            and other relevant details.
        """
        job_text_lower = job_text.lower()

        # Extract components
        actions = self._extract_actions(job_text_lower)
        deliverable = self._identify_deliverable(job_text_lower)
        constraints = self._extract_constraints(job_text)

        # Consolidate keywords
        keywords = list(set(actions + [deliverable] + list(constraints.keys())))

        return {
            'actions': actions,
            'deliverable': deliverable,
            'constraints': constraints,
            'keywords': keywords,
            'original_text': job_text
        }

    def _extract_actions(self, text: str) -> List[str]:
        """Extracts action verbs from the job description."""
        tokens = re.findall(r'\b\w+\b', text)
        return [verb for verb in self.action_verbs if verb in tokens]

    def _identify_deliverable(self, text: str) -> str:
        """Identifies the main deliverable from the job description."""
        for keyword in self.deliverable_keywords:
            if keyword in text:
                return keyword
        return 'report'  # Default deliverable

    def _extract_constraints(self, text: str) -> Dict:
        """Extracts various constraints from the job description."""
        constraints = {}

        # Time constraints
        time_match = re.search(self.constraints_patterns['time'], text, re.IGNORECASE)
        if time_match:
            constraints['time'] = f"{time_match.group(1)} {time_match.group(2)}"

        # Group size constraints
        group_match = re.search(self.constraints_patterns['group_size'], text, re.IGNORECASE)
        if group_match:
            constraints['group_size'] = f"{group_match.group(2)} {group_match.group(3)}"

        # Budget constraints
        budget_match = re.search(self.constraints_patterns['budget'], text, re.IGNORECASE)
        if budget_match:
            constraints['budget'] = budget_match.group(2)

        # Demographic constraints
        demographics_match = re.search(self.constraints_patterns['demographics'], text, re.IGNORECASE)
        if demographics_match:
            constraints['demographics'] = demographics_match.group(1)

        return constraints

# Example usage:
if __name__ == '__main__':
    job_parser = JobParser()

    test_job_1 = "Plan a trip of 4 days for a group of 10 college friends."
    parsed_job_1 = job_parser.parse_job(test_job_1)
    print("Test Job 1:", parsed_job_1)

    test_job_2 = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks."
    parsed_job_2 = job_parser.parse_job(test_job_2)
    print("Test Job 2:", parsed_job_2)

    test_job_3 = "Create and manage fillable forms for onboarding and compliance with a budget of $5000."
    parsed_job_3 = job_parser.parse_job(test_job_3)
    print("Test Job 3:", parsed_job_3)
