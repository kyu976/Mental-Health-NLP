"""
MindLog AI - Mental Health Journaling Application
Streamlit app for analyzing diary entries for burnout signals.
"""

import streamlit as st
from text_analyzer import (
    count_absolutist_words,
    count_first_person_pronouns,
    count_total_words,
    analyze_burnout_topics,
    get_top_burnout_topics,
    get_burnout_insight,
    calculate_burnout_risk
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
    Analyze your journal entries to detect potential burnout signals using topic modeling.
    Heavy weighting on topics found in r/burnout and r/depression (e.g., Work Overload, Sleep Disturbance, Emotional Exhaustion).
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
            
            # Topic modeling for burnout detection
            burnout_topics = analyze_burnout_topics(diary_entry)
            top_topics = get_top_burnout_topics(diary_entry, top_n=5)
            burnout_insight = get_burnout_insight(diary_entry)
            
            # Calculate direct burnout risk assessment
            burnout_assessment = calculate_burnout_risk(diary_entry)
            
            # Calculate frequencies (per 100 words)
            if total_words > 0:
                absolutist_frequency = (absolutist_count / total_words) * 100
                pronouns_frequency = (pronouns_count / total_words) * 100
            else:
                absolutist_frequency = 0
                pronouns_frequency = 0
            
            # Display Direct Burnout Analysis (Primary Feature)
            st.markdown("---")
            st.subheader("🔥 Burnout Risk Assessment")
            
            # Display risk score and level
            col_score1, col_score2 = st.columns([1, 2])
            with col_score1:
                # Color code based on risk level
                risk_colors = {
                    "Severe": "#FF0000",
                    "High": "#FF6B6B",
                    "Moderate": "#FFA500",
                    "Low": "#4CAF50"
                }
                risk_color = risk_colors.get(burnout_assessment["risk_level"], "#666666")
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; background-color: {risk_color}20; border-radius: 10px; border: 2px solid {risk_color};">
                        <h2 style="margin: 0; color: {risk_color};">{burnout_assessment['risk_score']:.0f}</h2>
                        <p style="margin: 5px 0 0 0; color: {risk_color}; font-weight: bold;">{burnout_assessment['risk_level']} Risk</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_score2:
                st.markdown(burnout_assessment["interpretation"])
            
            # Display recommendations
            if burnout_assessment["recommendations"]:
                st.markdown("#### 💡 Recommendations:")
                for i, rec in enumerate(burnout_assessment["recommendations"], 1):
                    st.markdown(f"{i}. {rec}")
            
            # Display Topic Modeling Section
            st.markdown("---")
            st.subheader("🔍 Topic Modeling Details")
            
            # Show topic modeling insight
            st.markdown(f"**{burnout_insight}**")
            
            # Display top burnout topics with correlation scores
            if top_topics and top_topics[0][1] > 0:
                st.markdown("#### Top Burnout Topic Correlations:")
                
                # Filter out topics with zero scores
                filtered_topics = [(topic_name, score) for topic_name, score in top_topics if score > 0]
                
                if filtered_topics:
                    # Create columns based on actual number of topics to display (max 3 columns)
                    num_cols = min(len(filtered_topics), 3)
                    cols = st.columns(num_cols)
                    
                    for idx, (topic_name, score) in enumerate(filtered_topics):
                        with cols[idx % num_cols]:
                            # Display as a chip/tag with score
                            percentage = score * 100
                            st.markdown(
                                f"""
                                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; color: #262730;">
                                    <strong style="color: #262730; font-size: 16px;">{topic_name}</strong><br>
                                    <small style="color: #505050; font-size: 14px;">Correlation: {percentage:.1f}%</small>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # Show detailed topic breakdown
                with st.expander("View All Topic Correlations"):
                    for topic_name, score in sorted(burnout_topics.items(), key=lambda x: x[1], reverse=True):
                        if score > 0:
                            st.progress(score, text=f"{topic_name}: {score*100:.1f}%")
            else:
                st.info("No significant burnout topic correlations detected.")
            
            # Display basic metrics
            st.markdown("---")
            st.subheader("📊 Basic Metrics")
            
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
            st.subheader("💡 Additional Insights")
            
            if absolutist_count > 0:
                st.info(
                    f"**Absolutist Language Detected**: Your entry contains {absolutist_count} "
                    f"absolutist word(s) ({absolutist_frequency:.2f} per 100 words). "
                    "High use of absolutist words (like 'always', 'never', 'completely') "
                    "may indicate stress or negative thinking patterns."
                )
            
            if pronouns_count > 0:
                st.info(
                    f"**Self-Reference**: Your entry contains {pronouns_count} "
                    f"first-person pronoun(s) ({pronouns_frequency:.2f} per 100 words). "
                    "This indicates how much you're focusing on yourself in your thoughts."
                )
                
        else:
            st.warning("Please enter some text before analyzing.")


if __name__ == "__main__":
    main()
