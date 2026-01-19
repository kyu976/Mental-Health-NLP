"""
Text analysis utilities for MindLog AI.
Functions to analyze text for burnout signals including absolutist words,
first-person pronoun usage, and topic modeling for burnout detection.
"""

import re
from typing import Dict, List, Tuple
from collections import Counter


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


# Burnout topic clusters based on Reddit patterns (r/burnout, r/depression, etc.)
BURNOUT_TOPICS = {
    "Work Overload": [
        'overwhelmed', 'overworked', 'exhausted', 'stressed', 'pressure',
        'deadline', 'workload', 'busy', 'swamped', 'drowning', 'crushed',
        'impossible', 'too much', 'can\'t keep up', 'falling behind',
        'burning out', 'burnout', 'tired', 'drained', 'empty'
    ],
    "Sleep Disturbance": [
        'sleep', 'insomnia', 'sleepless', 'restless', 'night', 'awake',
        'tossing', 'turning', 'nightmare', 'dream', 'wake up', 'can\'t sleep',
        'exhausted but can\'t sleep', 'fatigue', 'tired but wired',
        'sleeping problems', 'sleep issues'
    ],
    "Emotional Exhaustion": [
        'emotionally drained', 'empty', 'numb', 'feeling nothing',
        'detached', 'disconnected', 'apathetic', 'don\'t care',
        'lost interest', 'no motivation', 'unmotivated', 'depressed',
        'sad', 'hopeless', 'helpless', 'worthless', 'guilty'
    ],
    "Depersonalization": [
        'not myself', 'different person', 'don\'t recognize', 'lost',
        'identity', 'who am i', 'confused', 'foggy', 'brain fog',
        'can\'t think', 'memory', 'forgetful', 'spacing out'
    ],
    "Reduced Accomplishment": [
        'failure', 'failed', 'not good enough', 'incompetent', 'useless',
        'can\'t do anything', 'nothing works', 'stuck', 'no progress',
        'regressing', 'getting worse', 'declining', 'productivity',
        'efficiency', 'performance'
    ],
    "Physical Symptoms": [
        'headache', 'migraine', 'pain', 'ache', 'sore', 'tense',
        'muscle', 'back pain', 'neck pain', 'stomach', 'nausea',
        'dizzy', 'lightheaded', 'heart', 'chest', 'breathing',
        'shortness of breath', 'panic', 'anxiety attack'
    ],
    "Social Withdrawal": [
        'lonely', 'isolated', 'alone', 'withdraw', 'avoid', 'don\'t want to',
        'cancel', 'plans', 'social', 'friends', 'family', 'people',
        'don\'t want to talk', 'don\'t want to see', 'hermit', 'hide'
    ],
    "Cynicism": [
        'cynical', 'negative', 'pessimistic', 'bitter', 'resentful',
        'angry', 'frustrated', 'irritated', 'annoyed', 'hate',
        'despise', 'contempt', 'sarcastic', 'jaded'
    ]
}


def analyze_burnout_topics(text: str) -> Dict[str, float]:
    """
    Analyze text for burnout-related topics and return correlation scores.
    
    Based on topic modeling patterns found in Reddit burnout/depression communities.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary mapping topic names to correlation scores (0-1)
    """
    text_lower = text.lower()
    topic_scores = {}
    
    for topic_name, keywords in BURNOUT_TOPICS.items():
        matches = 0
        total_keywords = len(keywords)
        
        # Sort keywords by length (longest first) to match phrases before single words
        sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)
        
        for keyword in sorted_keywords:
            # Use word boundaries for single words, phrase matching for multi-word
            if ' ' in keyword:
                # Phrase matching - check if phrase exists in text
                if keyword in text_lower:
                    matches += 1
            else:
                # Word boundary matching for single words
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    matches += 1
        
        # Calculate correlation score (normalized to 0-1)
        # Using a weighted approach: more matches = higher correlation
        if total_keywords > 0:
            base_score = matches / total_keywords
            # Boost score if multiple keywords found (indicates stronger correlation)
            boost = min(matches * 0.1, 0.3)  # Max boost of 0.3
            correlation = min(base_score + boost, 1.0)
        else:
            correlation = 0.0
        
        topic_scores[topic_name] = correlation
    
    return topic_scores


def get_top_burnout_topics(text: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Get the top N burnout topics with highest correlation scores.
    
    Args:
        text: Input text to analyze
        top_n: Number of top topics to return
        
    Returns:
        List of tuples (topic_name, correlation_score) sorted by score descending
    """
    topic_scores = analyze_burnout_topics(text)
    # Sort by score descending and return top N
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:top_n]


def get_burnout_insight(text: str) -> str:
    """
    Generate a burnout insight message based on topic correlations.
    
    Similar to the example: "Your recent entries show a strong correlation 
    with the 'Sleep Disturbance' topic cluster."
    
    Args:
        text: Input text to analyze
        
    Returns:
        Insight message string
    """
    top_topics = get_top_burnout_topics(text, top_n=1)
    
    if not top_topics or top_topics[0][1] < 0.1:
        return "No strong burnout topic correlations detected in your entry."
    
    topic_name, score = top_topics[0]
    
    # Determine correlation strength
    if score >= 0.5:
        strength = "strong"
    elif score >= 0.3:
        strength = "moderate"
    else:
        strength = "some"
    
    return f"Your recent entries show a {strength} correlation with the '{topic_name}' topic cluster."


def calculate_burnout_risk(text: str) -> Dict[str, any]:
    """
    Calculate overall burnout risk score and provide direct analysis.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with burnout risk assessment including:
        - risk_score: float (0-100)
        - risk_level: str (Low, Moderate, High, Severe)
        - primary_concerns: list of top 3 topics
        - interpretation: str (direct analysis message)
        - recommendations: list of str
    """
    # Get topic correlations
    topic_scores = analyze_burnout_topics(text)
    
    # Get basic metrics
    absolutist_count = count_absolutist_words(text)
    pronouns_count = count_first_person_pronouns(text)
    total_words = count_total_words(text)
    
    # Calculate risk score components
    # 1. Topic correlation scores (weighted average of top 3)
    top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    avg_topic_score = sum(score for _, score in top_topics) / len(top_topics) if top_topics else 0
    
    # 2. Absolutist word frequency (normalized)
    absolutist_freq = (absolutist_count / total_words * 100) if total_words > 0 else 0
    absolutist_component = min(absolutist_freq / 5.0, 1.0) * 30  # Max 30 points, normalized to 5% frequency
    
    # 3. Number of active burnout topics (topics with score > 0.2)
    active_topics = sum(1 for score in topic_scores.values() if score > 0.2)
    topic_diversity_component = min(active_topics / 3.0, 1.0) * 20  # Max 20 points
    
    # 4. Overall topic correlation strength
    topic_strength_component = avg_topic_score * 50  # Max 50 points
    
    # Calculate total risk score (0-100)
    risk_score = min(absolutist_component + topic_diversity_component + topic_strength_component, 100)
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "Severe"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    # Get primary concerns (top 3 topics with scores > 0.1)
    primary_concerns = [
        topic for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        if score > 0.1
    ][:3]
    
    # Generate interpretation
    if risk_level == "Severe":
        interpretation = (
            f"⚠️ **Severe Burnout Risk Detected** (Score: {risk_score:.0f}/100)\n\n"
            f"Your journal entry shows multiple strong indicators of burnout. You're experiencing significant "
            f"stress across {len(primary_concerns)} key areas: {', '.join(primary_concerns[:2])}. "
            f"This level of burnout can significantly impact your physical and mental health. "
            f"Consider seeking professional support and making immediate changes to reduce stress."
        )
    elif risk_level == "High":
        interpretation = (
            f"🔴 **High Burnout Risk** (Score: {risk_score:.0f}/100)\n\n"
            f"Your entry indicates elevated burnout symptoms. Key concerns include: {', '.join(primary_concerns[:2])}. "
            f"You may be experiencing chronic stress that's affecting your well-being. "
            f"It's important to take proactive steps to manage these stressors before they worsen."
        )
    elif risk_level == "Moderate":
        interpretation = (
            f"🟡 **Moderate Burnout Risk** (Score: {risk_score:.0f}/100)\n\n"
            f"Your entry shows some burnout indicators, particularly around: {', '.join(primary_concerns[:1]) if primary_concerns else 'general stress'}. "
            f"While not severe, these patterns suggest you may benefit from stress management strategies "
            f"and self-care practices to prevent escalation."
        )
    else:
        interpretation = (
            f"🟢 **Low Burnout Risk** (Score: {risk_score:.0f}/100)\n\n"
            f"Your entry shows minimal burnout indicators. You appear to be managing stress relatively well. "
            f"Continue practicing self-care and monitoring your well-being."
        )
    
    # Generate recommendations based on primary concerns
    recommendations = []
    if "Work Overload" in primary_concerns:
        recommendations.append("Set clear boundaries between work and personal time")
        recommendations.append("Break large tasks into smaller, manageable steps")
        recommendations.append("Consider discussing workload with supervisor")
    if "Sleep Disturbance" in primary_concerns:
        recommendations.append("Establish a consistent sleep schedule")
        recommendations.append("Create a relaxing bedtime routine")
        recommendations.append("Limit screen time before bed")
    if "Emotional Exhaustion" in primary_concerns:
        recommendations.append("Practice mindfulness or meditation")
        recommendations.append("Engage in activities that bring you joy")
        recommendations.append("Consider speaking with a mental health professional")
    if "Social Withdrawal" in primary_concerns:
        recommendations.append("Make small efforts to connect with others")
        recommendations.append("Reach out to trusted friends or family")
        recommendations.append("Consider joining a support group")
    if "Physical Symptoms" in primary_concerns:
        recommendations.append("Schedule a check-up with your healthcare provider")
        recommendations.append("Practice relaxation techniques (deep breathing, yoga)")
        recommendations.append("Ensure you're getting regular physical activity")
    
    # Add general recommendations if none specific
    if not recommendations:
        recommendations.append("Continue monitoring your stress levels")
        recommendations.append("Maintain healthy routines and self-care practices")
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "primary_concerns": primary_concerns,
        "interpretation": interpretation,
        "recommendations": recommendations
    }
