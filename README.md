# Mental-Health-NLP

A sentiment analysis web application powered by DistilBERT that analyzes text and returns sentiment results (positive, negative, or neutral) with visual feedback.

## Features

- 🚀 Pre-trained DistilBERT model for sentiment analysis
- 🌐 Simple and intuitive web interface
- 🎨 Color-coded results (green for positive, red for negative)
- ⚡ Ready to run - no training required

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## Usage

1. Enter text in the text area
2. Click "Analyze Sentiment" or press Ctrl+Enter (Cmd+Enter on Mac)
3. View the sentiment result with confidence score

## Technology Stack

- **Backend:** Flask (Python)
- **ML Model:** DistilBERT (Hugging Face Transformers)
- **Frontend:** HTML, CSS, JavaScript
