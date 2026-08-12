"""
Query expansion for better retrieval recall.
Generates synonym variations of user queries to improve semantic search.
"""
from __future__ import annotations

import re
from typing import List


# Domain-specific synonym dictionary
SYNONYMS = {
    # GitLab/HR domain
    "hiring": ["recruitment", "talent acquisition", "interview process", "job application"],
    "remote work": ["distributed team", "work from home", "wfh", "telework"],
    "performance": ["performance review", "evaluation", "assessment", "talent review"],
    "onboarding": ["new hire process", "induction", "first day", "orientation"],
    "code review": ["merge request review", "MR review", "pull request review", "PR review"],
    "incident": ["outage", "service disruption", "production issue", "emergency"],
    "security": ["cybersecurity", "infosec", "data protection", "secure"],
    
    # Basecamp domain
    "vacation": ["time off", "PTO", "holiday", "sabbatical"],
    "meeting": ["sync", "standup", "check-in", "discussion"],
    "salary": ["compensation", "pay", "wage", "income"],
    "remote": ["work from anywhere", "distributed", "location independent"],
    
    # EduversePak/Education domain
    "attendance": ["presence", "roll call", "check-in"],
    "grade": ["mark", "score", "evaluation", "result"],
    "student": ["learner", "pupil", "enrolled user"],
    "teacher": ["instructor", "faculty", "educator"],
    "course": ["class", "subject", "module"],
    "session": ["term", "semester", "period"],
}


def expand_query(question: str) -> List[str]:
    """
    Generate query variations using synonym expansion.
    
    Args:
        question: Original user question
        
    Returns:
        List of query variants (original + expansions)
    """
    variants = [question]
    question_lower = question.lower()
    
    # Find matching synonyms
    for key, synonyms in SYNONYMS.items():
        if key in question_lower:
            # Create variants by replacing key with each synonym
            for synonym in synonyms[:2]:  # Limit to top 2 synonyms to avoid explosion
                variant = re.sub(
                    rf'\b{re.escape(key)}\b', 
                    synonym, 
                    question, 
                    flags=re.IGNORECASE
                )
                if variant != question:
                    variants.append(variant)
    
    return variants[:5]  # Max 5 variants


def get_expansion_terms(question: str) -> List[str]:
    """
    Get just the expansion terms without rewriting the question.
    Useful for query expansion in retrieval (not generation).
    """
    question_lower = question.lower()
    expansions = []
    
    for key, synonyms in SYNONYMS.items():
        if key in question_lower:
            expansions.extend(synonyms[:3])
    
    return expansions
