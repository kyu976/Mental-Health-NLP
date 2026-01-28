# MindLog AI 🧠

A mental health journaling application that analyzes user text to detect burnout signals using Natural Language Processing (NLP) techniques. MindLog AI helps users identify potential burnout indicators in their journal entries through linguistic pattern analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methods & Techniques](#methods--techniques)
- [Metrics](#metrics)
- [Burnout Risk Calculation](#burnout-risk-calculation)
- [Example Results](#example-results)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)

## 🎯 Overview

MindLog AI is a Streamlit-based web application that performs real-time analysis of journal entries to detect linguistic patterns associated with burnout. The application uses NLP techniques to identify:

- **Absolutist language patterns** (all-or-nothing thinking)
- **First-person pronoun usage** (self-focus indicators)
- **Overall burnout risk assessment** (composite score)

The tool is designed to help users become more aware of their stress levels and thinking patterns, providing actionable insights and recommendations.

## ✨ Features

- **Real-time Text Analysis**: Instant analysis of journal entries
- **Burnout Risk Assessment**: Comprehensive risk scoring (0-100) with four risk levels
- **Detailed Metrics**: Word frequency analysis and pattern detection
- **Personalized Recommendations**: Actionable suggestions based on analysis results
- **Modern UI**: Clean, calming interface designed for mental health applications

## 🔬 Methods & Techniques

### 1. **Pattern Matching with Regular Expressions**

The application uses Python's `re` module to identify specific linguistic patterns:

- **Word Boundary Matching**: Uses `\b` word boundaries to ensure whole-word matching only
- **Case-Insensitive Analysis**: Converts all text to lowercase for consistent matching
- **Exact Pattern Matching**: Prevents false positives from partial word matches

### 2. **Frequency Analysis**

Calculates normalized frequencies to account for varying entry lengths:

- **Per-100-Words Normalization**: Converts raw counts to percentages for fair comparison
- **Total Word Count**: Provides context for frequency calculations

### 3. **Composite Risk Scoring**

Combines multiple linguistic indicators into a single risk score:

- **Weighted Components**: Different weights for different indicators
- **Combination Factors**: Considers interactions between multiple indicators
- **Normalization**: Scales all components to a 0-100 range

## 📊 Metrics

### 1. **Total Words**
- **Description**: The total number of words in the journal entry
- **Purpose**: Provides context for frequency calculations and entry length analysis
- **Method**: Simple whitespace-based word splitting

### 2. **Absolutist Words**
- **Description**: Words that indicate all-or-nothing thinking patterns
- **Examples**: "always", "never", "completely", "nothing", "everything", "totally", "absolutely", "entirely", "fully", "perfect", "impossible", etc.
- **Detection**: Pattern matching against a predefined list of 28 absolutist words
- **Significance**: High frequency may suggest stress, anxiety, or negative cognitive patterns associated with burnout
- **Frequency Calculation**: `(absolutist_count / total_words) × 100`

### 3. **First-Person Pronouns**
- **Description**: Words that indicate self-focus and introspection
- **Examples**: "I", "me", "my", "myself", "mine"
- **Detection**: Pattern matching against a predefined list of 5 first-person pronouns
- **Significance**: While normal in journaling, excessive use combined with absolutist language can indicate heightened stress, rumination, or burnout symptoms
- **Frequency Calculation**: `(pronouns_count / total_words) × 100`

### 4. **Burnout Risk Score**
- **Description**: Composite score (0-100) indicating overall burnout risk
- **Risk Levels**:
  - **Low** (0-29): Minimal burnout indicators
  - **Moderate** (30-49): Some burnout indicators present
  - **High** (50-69): Elevated burnout symptoms
  - **Severe** (70-100): Strong indicators of burnout

## 🧮 Burnout Risk Calculation

The burnout risk score is calculated using a weighted formula:

### Score Components

1. **Absolutist Word Component** (Max 40 points)
   - Formula: `min(absolutist_freq / 5.0, 1.0) × 40`
   - Normalized to 5% frequency threshold
   - Higher absolutist word frequency increases this component

2. **First-Person Pronoun Component** (Max 30 points)
   - Formula: `min(pronouns_freq / 10.0, 1.0) × 30`
   - Normalized to 10% frequency threshold
   - Higher pronoun frequency increases this component

3. **Combination Factor** (Max 30 points)
   - **High Combination** (+30 points): `absolutist_freq > 3%` AND `pronouns_freq > 5%`
   - **Moderate Combination** (+15 points): `absolutist_freq > 2%` OR `pronouns_freq > 4%`
   - **No Boost** (0 points): Otherwise
   - Accounts for the interaction effect when both indicators are present

### Final Score

```
Risk Score = min(absolutist_component + pronouns_component + combination_boost, 100)
```

The score is capped at 100 to ensure it stays within the defined range.

## 📈 Example Results

### Example 1: Low Risk Entry

**Input Text:**
```
Today was a productive day. I completed my tasks and felt satisfied with the progress. 
The weather was nice, so I took a walk during lunch. I'm looking forward to the weekend.
```

**Results:**
- **Total Words**: 28
- **Absolutist Words**: 0 (0.00 per 100 words)
- **First-Person Pronouns**: 3 (10.71 per 100 words)
- **Burnout Risk Score**: 12/100
- **Risk Level**: Low
- **Interpretation**: Your entry shows minimal burnout indicators. You're using 0 absolutist word(s) (0.0% frequency) and 3 first-person pronoun(s) (10.7% frequency). You appear to be managing stress relatively well.

### Example 2: Moderate Risk Entry

**Input Text:**
```
I feel completely overwhelmed with everything on my plate. I never seem to have enough time, 
and my work is always piling up. I'm worried I can't handle all of this.
```

**Results:**
- **Total Words**: 30
- **Absolutist Words**: 4 (13.33 per 100 words)
  - Found: "completely" (1x), "everything" (1x), "never" (1x), "always" (1x)
- **First-Person Pronouns**: 4 (13.33 per 100 words)
  - Found: "I" (4x)
- **Burnout Risk Score**: 45/100
- **Risk Level**: Moderate
- **Interpretation**: Your entry shows some burnout indicators. You're using 4 absolutist word(s) (13.3% frequency) and 4 first-person pronoun(s) (13.3% frequency). While not severe, these patterns suggest you may benefit from stress management strategies and self-care practices to prevent escalation.
- **Recommendations**:
  - Practice reframing absolutist thoughts (e.g., 'always' → 'often', 'never' → 'rarely')
  - Consider cognitive behavioral techniques to challenge all-or-nothing thinking
  - Maintain healthy routines and self-care practices

### Example 3: High Risk Entry

**Input Text:**
```
I'm absolutely exhausted. Nothing I do seems to work anymore. I feel like I'm completely 
failing at everything. My life is totally out of control. I can't handle this anymore. 
I never get a break, and I'm always stressed. Everything feels impossible.
```

**Results:**
- **Total Words**: 42
- **Absolutist Words**: 8 (19.05 per 100 words)
  - Found: "absolutely" (1x), "nothing" (1x), "completely" (1x), "everything" (1x), "totally" (1x), "never" (1x), "always" (1x), "impossible" (1x)
- **First-Person Pronouns**: 6 (14.29 per 100 words)
  - Found: "I" (6x)
- **Burnout Risk Score**: 68/100
- **Risk Level**: High
- **Interpretation**: Your entry indicates elevated burnout symptoms. You're using 8 absolutist word(s) (19.0% frequency) and 6 first-person pronoun(s) (14.3% frequency). You may be experiencing chronic stress that's affecting your well-being. It's important to take proactive steps to manage these stressors before they worsen.
- **Recommendations**:
  - Practice reframing absolutist thoughts
  - Consider cognitive behavioral techniques
  - Consider speaking with a mental health professional
  - Practice relaxation techniques (meditation, deep breathing, yoga)
  - Set clear boundaries between work and personal time

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Mental-Health-NLP
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   
   Or use the provided script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

4. **Access the application:**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## 💻 Usage

1. **Enter Your Journal Entry**: Type or paste your journal entry into the text area
2. **Click "Analyze"**: The application will process your text in real-time
3. **Review Results**:
   - **Burnout Risk Assessment**: View your risk score and level
   - **Analysis Results**: See detailed metrics for total words, absolutist words, and first-person pronouns
   - **Additional Insights**: View specific words/pronouns found and their frequencies
   - **Recommendations**: Read personalized suggestions based on your results

## 🛠 Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3
- **NLP Library**: Python `re` (Regular Expressions) for pattern matching
- **Text Processing**: Built-in Python string methods

### Dependencies

- `streamlit>=1.28.0`: Web application framework
- `nltk>=3.8`: Natural Language Toolkit (for potential future enhancements)

## 📁 Project Structure

```
Mental-Health-NLP/
├── app.py                 # Main Streamlit application
├── text_analyzer.py        # Core NLP analysis functions
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # License information
└── twitter train data.csv  # Training data (for reference)
```

### Key Files

- **`app.py`**: Contains the Streamlit UI, user interface logic, and result display
- **`text_analyzer.py`**: Contains all NLP analysis functions:
  - `count_absolutist_words()`: Counts absolutist words
  - `count_first_person_pronouns()`: Counts first-person pronouns
  - `count_total_words()`: Counts total words
  - `get_absolutist_words_found()`: Returns dictionary of found absolutist words
  - `get_first_person_pronouns_found()`: Returns dictionary of found pronouns
  - `calculate_burnout_risk()`: Calculates composite burnout risk score

## 🔍 How It Works

1. **Text Input**: User enters journal entry via Streamlit text area
2. **Text Processing**: 
   - Text is converted to lowercase for case-insensitive matching
   - Words are extracted using whitespace splitting
3. **Pattern Matching**:
   - Regular expressions with word boundaries match predefined word lists
   - Counts are aggregated for each category
4. **Frequency Calculation**:
   - Raw counts are normalized to per-100-words frequencies
5. **Risk Scoring**:
   - Components are calculated and combined
   - Risk level is determined based on score thresholds
6. **Result Generation**:
   - Interpretation text is generated based on risk level
   - Recommendations are created based on detected patterns
7. **Display**: Results are rendered in the Streamlit interface with styled HTML

## ⚠️ Important Notes

- **Not a Medical Tool**: This application is for informational and self-awareness purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
- **Privacy**: All text analysis is performed locally. No data is sent to external servers.
- **Limitations**: The analysis is based on linguistic patterns and may not capture all aspects of mental health or burnout.

## 📝 Future Enhancements

Potential improvements for future versions:

- Integration with machine learning models for more sophisticated analysis
- Topic modeling to identify specific burnout-related themes
- Historical tracking of entries over time
- Export functionality for results
- Integration with sentiment analysis models
- Multi-language support

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Disclaimer**: This tool is designed for self-awareness and educational purposes. If you're experiencing severe burnout or mental health issues, please consult with a qualified mental health professional.
