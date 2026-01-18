"""
MindLog AI - Mental Health Journaling Application
Streamlit app for analyzing diary entries for burnout signals.
"""

import streamlit as st
from text_analyzer import (
    count_absolutist_words,
    count_first_person_pronouns,
    count_total_words
)


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="MindLog AI",
        page_icon="🧠",
        layout="wide"
    )
    
    # Title and description
    st.title("🧠 MindLog AI - Mental Health Journaling")
    st.markdown("""
    Analyze your journal entries to detect potential burnout signals.
    Enter your thoughts below and click 'Analyze' to see insights.
    """)
    
    # Text input area
    diary_entry = st.text_area(
        "Write your diary entry here:",
        height=200,
        placeholder="Today I feel completely overwhelmed. I never seem to have enough time, and everything feels impossible..."
    )
    
    # Analyze button
    if st.button("Analyze", type="primary"):
        if diary_entry.strip():
            # Perform analysis
            total_words = count_total_words(diary_entry)
            absolutist_count = count_absolutist_words(diary_entry)
            pronouns_count = count_first_person_pronouns(diary_entry)
            
            # Calculate frequencies (per 100 words)
            if total_words > 0:
                absolutist_frequency = (absolutist_count / total_words) * 100
                pronouns_frequency = (pronouns_count / total_words) * 100
            else:
                absolutist_frequency = 0
                pronouns_frequency = 0
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Create columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Words", total_words)
            
            with col2:
                st.metric(
                    "Absolutist Words",
                    absolutist_count,
                    delta=f"{absolutist_frequency:.2f} per 100 words"
                )
            
            with col3:
                st.metric(
                    "First-Person Pronouns",
                    pronouns_count,
                    delta=f"{pronouns_frequency:.2f} per 100 words"
                )
            
            # Additional insights section
            st.markdown("---")
            st.subheader("💡 Insights")
            
            if absolutist_count > 0:
                st.info(
                    f"**Absolutist Language Detected**: Your entry contains {absolutist_count} "
                    f"absolutist word(s) ({absolutist_frequency:.2f} per 100 words). "
                    "High use of absolutist words (like 'always', 'never', 'completely') "
                    "may indicate stress or negative thinking patterns."
                )
            else:
                st.success("No absolutist words detected in your entry.")
            
            if pronouns_count > 0:
                st.info(
                    f"**Self-Reference**: Your entry contains {pronouns_count} "
                    f"first-person pronoun(s) ({pronouns_frequency:.2f} per 100 words). "
                    "This indicates how much you're focusing on yourself in your thoughts."
                )
            else:
                st.info("No first-person pronouns detected in your entry.")
                
        else:
            st.warning("Please enter some text before analyzing.")


if __name__ == "__main__":
    main()
