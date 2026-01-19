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


def get_absolutist_words_found(text: str) -> list:
    """
    Get the list of absolutist words found in the text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of absolutist words found (with counts)
    """
    absolutist_words = [
        'always', 'never', 'completely', 'nothing', 'everything',
        'totally', 'absolutely', 'entirely', 'fully', 'whole',
        'all', 'none', 'nobody', 'no one', 'nowhere', 'everywhere',
        'everyone', 'everybody', 'anyone', 'anybody', 'anything',
        'anywhere', 'perfect', 'perfectly', 'impossible', 'impossibly',
        'forever', 'ever'
    ]
    
    text_lower = text.lower()
    found_words = {}
    
    for word in absolutist_words:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found_words[word] = len(matches)
    
    return found_words


def get_first_person_pronouns_found(text: str) -> dict:
    """
    Get the list of first-person pronouns found in the text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary of pronouns found with counts
    """
    first_person_pronouns = ['i', 'me', 'my', 'myself', 'mine']
    text_lower = text.lower()
    found_pronouns = {}
    
    for pronoun in first_person_pronouns:
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found_pronouns[pronoun] = len(matches)
    
    return found_pronouns


def calculate_burnout_risk(text: str) -> dict:
    """
    Calculate overall burnout risk score and provide direct analysis.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with burnout risk assessment including:
        - risk_score: float (0-100)
        - risk_level: str (Low, Moderate, High, Severe)
        - interpretation: str (direct analysis message)
        - recommendations: list of str
    """
    # Get basic metrics
    absolutist_count = count_absolutist_words(text)
    pronouns_count = count_first_person_pronouns(text)
    total_words = count_total_words(text)
    
    # Calculate risk score components
    # 1. Absolutist word frequency (normalized)
    absolutist_freq = (absolutist_count / total_words * 100) if total_words > 0 else 0
    absolutist_component = min(absolutist_freq / 5.0, 1.0) * 40  # Max 40 points, normalized to 5% frequency
    
    # 2. First-person pronoun frequency (indicates self-focus, which can correlate with stress)
    pronouns_freq = (pronouns_count / total_words * 100) if total_words > 0 else 0
    pronouns_component = min(pronouns_freq / 10.0, 1.0) * 30  # Max 30 points, normalized to 10% frequency
    
    # 3. Combination factor (both high = higher risk)
    if absolutist_freq > 3 and pronouns_freq > 5:
        combination_boost = 30
    elif absolutist_freq > 2 or pronouns_freq > 4:
        combination_boost = 15
    else:
        combination_boost = 0
    
    # Calculate total risk score (0-100)
    risk_score = min(absolutist_component + pronouns_component + combination_boost, 100)
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "Severe"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    # Generate interpretation (using HTML for better formatting)
    if risk_level == "Severe":
        interpretation = (
            f'<p style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 700; font-size: 1.1rem;">'
            f'⚠️ Severe Burnout Risk Detected (Score: {risk_score:.0f}/100)</p>'
            f'<p style="margin: 0; color: #1e293b; line-height: 1.7;">'
            f'Your journal entry shows multiple strong indicators of burnout. '
            f'You\'re using <strong>{absolutist_count}</strong> absolutist word(s) ({absolutist_freq:.1f}% frequency) and '
            f'<strong>{pronouns_count}</strong> first-person pronoun(s) ({pronouns_freq:.1f}% frequency). '
            f'This combination suggests significant stress and negative thinking patterns. '
            f'Consider seeking professional support and making immediate changes to reduce stress.</p>'
        )
    elif risk_level == "High":
        interpretation = (
            f'<p style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 700; font-size: 1.1rem;">'
            f'🔴 High Burnout Risk (Score: {risk_score:.0f}/100)</p>'
            f'<p style="margin: 0; color: #1e293b; line-height: 1.7;">'
            f'Your entry indicates elevated burnout symptoms. '
            f'You\'re using <strong>{absolutist_count}</strong> absolutist word(s) ({absolutist_freq:.1f}% frequency) and '
            f'<strong>{pronouns_count}</strong> first-person pronoun(s) ({pronouns_freq:.1f}% frequency). '
            f'You may be experiencing chronic stress that\'s affecting your well-being. '
            f'It\'s important to take proactive steps to manage these stressors before they worsen.</p>'
        )
    elif risk_level == "Moderate":
        interpretation = (
            f'<p style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 700; font-size: 1.1rem;">'
            f'🟡 Moderate Burnout Risk (Score: {risk_score:.0f}/100)</p>'
            f'<p style="margin: 0; color: #1e293b; line-height: 1.7;">'
            f'Your entry shows some burnout indicators. '
            f'You\'re using <strong>{absolutist_count}</strong> absolutist word(s) ({absolutist_freq:.1f}% frequency) and '
            f'<strong>{pronouns_count}</strong> first-person pronoun(s) ({pronouns_freq:.1f}% frequency). '
            f'While not severe, these patterns suggest you may benefit from stress management strategies '
            f'and self-care practices to prevent escalation.</p>'
        )
    else:
        interpretation = (
            f'<p style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 700; font-size: 1.1rem;">'
            f'🟢 Low Burnout Risk (Score: {risk_score:.0f}/100)</p>'
            f'<p style="margin: 0; color: #1e293b; line-height: 1.7;">'
            f'Your entry shows minimal burnout indicators. '
            f'You\'re using <strong>{absolutist_count}</strong> absolutist word(s) ({absolutist_freq:.1f}% frequency) and '
            f'<strong>{pronouns_count}</strong> first-person pronoun(s) ({pronouns_freq:.1f}% frequency). '
            f'You appear to be managing stress relatively well. '
            f'Continue practicing self-care and monitoring your well-being.</p>'
        )
    
    # Generate recommendations
    recommendations = []
    if absolutist_count > 3:
        recommendations.append("Practice reframing absolutist thoughts (e.g., 'always' → 'often', 'never' → 'rarely')")
        recommendations.append("Consider cognitive behavioral techniques to challenge all-or-nothing thinking")
    if pronouns_count > 10:
        recommendations.append("Try to balance self-reflection with external perspectives")
        recommendations.append("Engage in activities that shift focus outward (helping others, nature walks)")
    if risk_level in ["High", "Severe"]:
        recommendations.append("Consider speaking with a mental health professional")
        recommendations.append("Practice relaxation techniques (meditation, deep breathing, yoga)")
        recommendations.append("Set clear boundaries between work and personal time")
    if risk_level == "Moderate":
        recommendations.append("Maintain healthy routines and self-care practices")
        recommendations.append("Monitor your stress levels regularly")
    
    # Add general recommendations if none specific
    if not recommendations:
        recommendations.append("Continue monitoring your stress levels")
        recommendations.append("Maintain healthy routines and self-care practices")
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "interpretation": interpretation,
        "recommendations": recommendations
    }
