"""
MindLog AI - Mental Health Journaling Application
Streamlit app for analyzing diary entries for burnout signals.
"""

import streamlit as st
from text_analyzer import (
    count_absolutist_words,
    count_first_person_pronouns,
    count_total_words,
    get_absolutist_words_found,
    get_first_person_pronouns_found,
    calculate_burnout_risk
)


def main():
    """Main Streamlit application."""
    # Calming color palette - white interface with gradient accents
    # Subtle background gradients (fading to white)
    BG_GRADIENT_1 = "linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%)"  # Light blue to white
    BG_GRADIENT_2 = "linear-gradient(180deg, #f3e5f5 0%, #ffffff 100%)"  # Light purple to white
    
    # Colorful gradient accents for UI elements
    PRIMARY_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)"  # Purple to pink
    ACCENT_GRADIENT_1 = "linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%)"  # Green
    ACCENT_GRADIENT_2 = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"  # Blue
    ACCENT_GRADIENT_3 = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"  # Pink/Red
    ACCENT_GRADIENT_4 = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"  # Purple
    ACCENT_GRADIENT_5 = "linear-gradient(135deg, #c5e3f6 0%, #a8d5e2 100%)"  # Light blue
    
    # Metric card gradients
    METRIC_GRADIENT_1 = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"  # Purple
    METRIC_GRADIENT_2 = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"  # Blue
    METRIC_GRADIENT_3 = "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"  # Green
    
    # Page configuration
    st.set_page_config(
        page_title="MindLog AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for white interface with gradient accents
    st.markdown(f"""
    <style>
    .main {{
        background: {BG_GRADIENT_1};
        padding: 2rem;
    }}
    .stApp {{
        background: {BG_GRADIENT_1};
    }}
    .stApp > div {{
        background: transparent;
    }}
    .block-container {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    [data-testid="stAppViewContainer"] {{
        background: {BG_GRADIENT_1};
    }}
    h1 {{
        color: #2d3748;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    .stButton>button {{
        background: {PRIMARY_GRADIENT};
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }}
    .stTextArea>div>div>textarea {{
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-radius: 10px;
        border: 2px solid rgba(79, 172, 254, 0.4);
        font-size: 16px;
        line-height: 1.6;
        font-weight: 400;
    }}
    .stTextArea>div>div>textarea::placeholder {{
        color: #94a3b8 !important;
        opacity: 0.8;
    }}
    .stTextArea>div>div>textarea:focus {{
        border-color: #4facfe;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.15);
        outline: none;
    }}
    /* Center content in first column of risk assessment */
    div[data-testid="column"]:first-of-type {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    /* Remove any hyperlink styling from risk score box */
    div[data-testid="column"]:first-of-type a {{
        text-decoration: none !important;
        color: inherit !important;
    }}
    div[data-testid="column"]:first-of-type a:hover {{
        text-decoration: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description with vibrant gradient styling
    st.markdown(f"""
    <div style="background: {PRIMARY_GRADIENT}; 
                padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;">
        <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.8rem; font-weight: 700; position: relative; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">🧠 MindLog AI</h1>
        <p style="color: rgba(255,255,255,0.95); text-align: center; margin: 0.8rem 0 0 0; font-size: 1.2rem; position: relative; font-weight: 400;">
            Mental Health Journaling & Burnout Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid; border-image: {ACCENT_GRADIENT_2}; border-image-slice: 1; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(79, 172, 254, 0.15);">
        <p style="margin: 0; color: #2d3748; font-size: 1.05rem; line-height: 1.7; font-weight: 500;">
            ✨ <strong>Analyze your journal entries</strong> to detect potential burnout signals using topic modeling.<br>
            📊 Heavy weighting on topics found in r/burnout and r/depression (e.g., Work Overload, Sleep Disturbance, Emotional Exhaustion).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input area with better styling
    st.markdown("""
    <style>
    /* Make sure label is visible */
    label[data-testid="stTextArea"] {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    label[data-testid="stTextArea"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    /* Target Streamlit's label wrapper */
    div[data-testid="stTextArea"] > label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }
    /* Ensure textarea text is visible */
    [data-baseweb="textarea"] {
        color: #1e293b !important;
    }
    textarea {
        color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add visible label above textarea
    st.markdown("""
    <div style="margin-bottom: 0.75rem;">
        <label style="color: #1e293b; font-weight: 600; font-size: 1.05rem; display: block;">
            Write your diary entry here:
        </label>
    </div>
    """, unsafe_allow_html=True)
    
    diary_entry = st.text_area(
        "",  # Empty label since we're adding our own
        height=200,
        placeholder="Today I feel completely overwhelmed. I never seem to have enough time, and everything feels impossible...",
        help="Enter your thoughts and feelings. The analysis will help identify burnout signals.",
        label_visibility="collapsed"  # Hide Streamlit's default label
    )
    
    # Analyze button
    if st.button("Analyze", type="primary"):
        if diary_entry.strip():
            # Perform analysis
            total_words = count_total_words(diary_entry)
            absolutist_count = count_absolutist_words(diary_entry)
            pronouns_count = count_first_person_pronouns(diary_entry)
            absolutist_words_found = get_absolutist_words_found(diary_entry)
            pronouns_found = get_first_person_pronouns_found(diary_entry)
            
            # Calculate burnout risk assessment
            burnout_assessment = calculate_burnout_risk(diary_entry)
            
            # Calculate frequencies (per 100 words)
            if total_words > 0:
                absolutist_frequency = (absolutist_count / total_words) * 100
                pronouns_frequency = (pronouns_count / total_words) * 100
            else:
                absolutist_frequency = 0
                pronouns_frequency = 0
            
            # Display Burnout Risk Assessment first
            st.markdown("---")
            st.markdown(f"""
            <div style="background: {PRIMARY_GRADIENT}; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
                <h2 style="color: white; text-align: center; margin: 0 0 1rem 0; font-weight: 700; font-size: 1.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    🔥 Burnout Risk Assessment
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display risk score and level
            col_score1, col_score2 = st.columns([1, 2])
            with col_score1:
                # Color code based on risk level
                risk_colors = {
                    "Severe": {"bg": "linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)", "text": "#ffffff", "border": "#ff4757"},
                    "High": {"bg": "linear-gradient(135deg, #ffa726 0%, #fb8c00 100%)", "text": "#ffffff", "border": "#ff9800"},
                    "Moderate": {"bg": "linear-gradient(135deg, #ffd54f 0%, #ffc107 100%)", "text": "#2d3748", "border": "#ffb300"},
                    "Low": {"bg": "linear-gradient(135deg, #81c784 0%, #66bb6a 100%)", "text": "#ffffff", "border": "#4caf50"}
                }
                risk_style = risk_colors.get(burnout_assessment["risk_level"], {"bg": "#e0e0e0", "text": "#2d3748", "border": "#9e9e9e"})
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center;">
                        <div style="padding: 30px 20px; background: {risk_style['bg']}; 
                                    border-radius: 15px; border: 3px solid {risk_style['border']}; 
                                    box-shadow: 0 8px 16px rgba(0,0,0,0.2); width: 100%; max-width: 100%;
                                    display: flex; flex-direction: column; align-items: center; justify-content: center;">
                            <div style="margin: 0; color: {risk_style['text']}; font-size: 3.5rem; font-weight: bold; line-height: 1; text-align: center;">
                                {burnout_assessment['risk_score']:.0f}
                            </div>
                            <div style="margin: 10px 0 0 0; color: {risk_style['text']}; font-weight: bold; font-size: 1.2rem; text-align: center;">
                                {burnout_assessment['risk_level']} Risk
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_score2:
                # Render interpretation with HTML styling
                st.markdown(
                    f"""
                    <div style="background: white; 
                                padding: 1.5rem; border-radius: 10px; border-left: 5px solid #764ba2; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        {burnout_assessment["interpretation"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display recommendations
            if burnout_assessment["recommendations"]:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                           padding: 0.75rem; border-radius: 10px; margin-top: 1rem; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #1e293b; margin: 0; font-weight: 700; font-size: 1.2rem;">
                        💡 Recommendations
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                rec_colors = ["#ff6b9d", "#c44569", "#f8b500", "#ffa07a", "#ff7675"]
                for i, rec in enumerate(burnout_assessment["recommendations"], 1):
                    color = rec_colors[i % len(rec_colors)]
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%); 
                                    padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                                    border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <span style="color: {color}; font-weight: bold; font-size: 1.1rem;">{i}.</span>
                            <span style="color: #2d3748; margin-left: 0.5rem;">{rec}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display Analysis Results with vibrant gradient header
            st.markdown("---")
            st.markdown(f"""
            <div style="background: {ACCENT_GRADIENT_2}; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(79, 172, 254, 0.25);">
                <h2 style="color: white; text-align: center; margin: 0; font-weight: 700; font-size: 1.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    📊 Analysis Results
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Modern list-style layout instead of cards
            metric_colors = [METRIC_GRADIENT_1, METRIC_GRADIENT_2, METRIC_GRADIENT_3]
            
            # Render metrics in a clean list format
            st.markdown(
                f"""
                <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 0; border-bottom: 1px solid rgba(0,0,0,0.08);">
                        <div style="display: flex; align-items: center; gap: 1.25rem; flex: 1;">
                            <div style="background: {metric_colors[0]}; width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); flex-shrink: 0;">
                                <span style="color: white; font-size: 1.8rem;">📝</span>
                            </div>
                            <div style="flex: 1; padding-right: 2rem; max-width: 70%;">
                                <h3 style="margin: 0 0 0.3rem 0; font-size: 1.2rem; color: #000000; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Total Words</h3>
                                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #64748b; font-weight: 400; line-height: 1.5;">The total number of words in your journal entry. Longer entries provide more context for analysis.</p>
                                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; font-weight: 500;">Entry length: {total_words} words</p>
                            </div>
                        </div>
                        <div style="text-align: right; margin-left: 1rem;">
                            <h2 style="margin: 0; font-size: 3rem; color: #2d3748; font-weight: 700; line-height: 1;">{total_words}</h2>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 0; border-bottom: 1px solid rgba(0,0,0,0.08);">
                        <div style="display: flex; align-items: center; gap: 1.25rem; flex: 1;">
                            <div style="background: {metric_colors[1]}; width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); flex-shrink: 0;">
                                <span style="color: white; font-size: 1.8rem;">🔴</span>
                            </div>
                            <div style="flex: 1; padding-right: 2rem; max-width: 70%;">
                                <h3 style="margin: 0 0 0.3rem 0; font-size: 1.2rem; color: #000000; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Absolutist Words</h3>
                                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #64748b; font-weight: 400; line-height: 1.5;">Words like "always", "never", "completely" that indicate all-or-nothing thinking. High frequency may suggest stress or burnout.</p>
                                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; font-weight: 500;">{absolutist_frequency:.2f} per 100 words</p>
                            </div>
                        </div>
                        <div style="text-align: right; margin-left: 1rem;">
                            <h2 style="margin: 0; font-size: 3rem; color: #2d3748; font-weight: 700; line-height: 1;">{absolutist_count}</h2>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 0;">
                        <div style="display: flex; align-items: center; gap: 1.25rem; flex: 1;">
                            <div style="background: {metric_colors[2]}; width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); flex-shrink: 0;">
                                <span style="color: white; font-size: 1.8rem;">👤</span>
                            </div>
                            <div style="flex: 1; padding-right: 2rem; max-width: 70%;">
                                <h3 style="margin: 0 0 0.3rem 0; font-size: 1.2rem; color: #000000; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">First-Person Pronouns</h3>
                                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #64748b; font-weight: 400; line-height: 1.5;">Words like "I", "me", "my", "myself", "mine" that indicate self-focus. Excessive use combined with absolutist language can indicate stress or burnout.</p>
                                <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; font-weight: 500;">{pronouns_frequency:.2f} per 100 words</p>
                            </div>
                        </div>
                        <div style="text-align: right; margin-left: 1rem;">
                            <h2 style="margin: 0; font-size: 3rem; color: #2d3748; font-weight: 700; line-height: 1;">{pronouns_count}</h2>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Additional insights section with vibrant gradient header
            st.markdown("---")
            st.markdown(f"""
            <div style="background: {ACCENT_GRADIENT_3}; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(225, 190, 231, 0.25);">
                <h2 style="color: white; text-align: center; margin: 0; font-weight: 700; font-size: 1.8rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    💡 Additional Insights
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            if absolutist_count > 0:
                words_list = ", ".join([f"{word} ({count}x)" for word, count in list(absolutist_words_found.items())[:10]])
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                                border-left: 5px solid; border-image: {ACCENT_GRADIENT_2}; border-image-slice: 1;
                                box-shadow: 0 4px 12px rgba(79, 172, 254, 0.15);">
                        <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                            <div style="background: {ACCENT_GRADIENT_2}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; box-shadow: 0 2px 8px rgba(79, 172, 254, 0.3);">
                                <span style="color: white; font-size: 1.3rem;">💭</span>
                            </div>
                            <strong style="color: #2d3748; font-size: 1.1rem;">Absolutist Language Detected</strong>
                        </div>
                        <p style="margin: 0 0 0.8rem 0; color: #2d3748; font-size: 1rem; line-height: 1.7; padding-left: 3.5rem;">
                            Your entry contains <strong>{absolutist_count}</strong> absolutist word(s) ({absolutist_frequency:.2f} per 100 words). 
                            High use of absolutist words (like 'always', 'never', 'completely') 
                            may indicate stress or negative thinking patterns.
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.9rem; padding-left: 3.5rem; font-style: italic;">
                            Found: {words_list if absolutist_words_found else "None"}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                                border-left: 5px solid #43e97b; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <p style="margin: 0; color: #2d3748; font-size: 1rem; line-height: 1.7;">
                            ✅ <strong>No absolutist words detected</strong> - Your language shows balanced thinking patterns.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            if pronouns_count > 0:
                pronouns_list = ", ".join([f"{pronoun} ({count}x)" for pronoun, count in list(pronouns_found.items())[:10]])
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                                border-left: 5px solid; border-image: {ACCENT_GRADIENT_1}; border-image-slice: 1;
                                box-shadow: 0 4px 12px rgba(67, 233, 123, 0.15);">
                        <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                            <div style="background: {ACCENT_GRADIENT_1}; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; box-shadow: 0 2px 8px rgba(67, 233, 123, 0.3);">
                                <span style="color: white; font-size: 1.3rem;">👤</span>
                            </div>
                            <strong style="color: #2d3748; font-size: 1.1rem;">Self-Reference</strong>
                        </div>
                        <p style="margin: 0 0 0.8rem 0; color: #2d3748; font-size: 1rem; line-height: 1.7; padding-left: 3.5rem;">
                            Your entry contains <strong>{pronouns_count}</strong> first-person pronoun(s) ({pronouns_frequency:.2f} per 100 words). 
                            This indicates how much you're focusing on yourself in your thoughts.
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.9rem; padding-left: 3.5rem; font-style: italic;">
                            Found: {pronouns_list if pronouns_found else "None"}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                                border-left: 5px solid #4facfe; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <p style="margin: 0; color: #2d3748; font-size: 1rem; line-height: 1.7;">
                            ℹ️ <strong>No first-person pronouns detected</strong> - Your entry uses minimal self-reference.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        else:
            st.warning("Please enter some text before analyzing.")


if __name__ == "__main__":
    main()
