"""
Text analysis utilities for MindLog AI.
Functions to analyze text for burnout signals including absolutist words
and first-person pronoun usage.
"""

import re


def count_absolutist_words(text: str) -> int:
    """
    Count occurrences of absolutist words in the given text.
    
    Absolutist words include: always, never, completely, nothing, everything,
    totally, absolutely, entirely, fully, whole, all, none, nobody, no one,
    nowhere, etc.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Count of absolutist words found
    """
    # List of absolutist words to detect
    absolutist_words = [
        'always', 'never', 'completely', 'nothing', 'everything',
        'totally', 'absolutely', 'entirely', 'fully', 'whole',
        'all', 'none', 'nobody', 'no one', 'nowhere', 'everywhere',
        'everyone', 'everybody', 'anyone', 'anybody', 'anything',
        'anywhere', 'perfect', 'perfectly', 'impossible', 'impossibly',
        'forever', 'never', 'ever'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences using word boundaries to avoid partial matches
    count = 0
    for word in absolutist_words:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    
    return count


def count_first_person_pronouns(text: str) -> int:
    """
    Count occurrences of first-person pronouns in the given text.
    
    First-person pronouns include: I, me, my, myself, mine.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Count of first-person pronouns found
    """
    # List of first-person pronouns
    first_person_pronouns = ['i', 'me', 'my', 'myself', 'mine']
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences using word boundaries
    count = 0
    for pronoun in first_person_pronouns:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    
    return count


def count_total_words(text: str) -> int:
    """
    Count total words in the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Total word count
    """
    # Split by whitespace and filter out empty strings
    words = text.split()
    return len(words)
