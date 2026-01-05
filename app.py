from flask import Flask, render_template, request, jsonify
import numpy as np
from transformers import pipeline

app = Flask(__name__)

# Lazy initialization of the sentiment analysis pipeline
# This will download the model on first use
sentiment_pipeline = None

def get_sentiment_pipeline():
    """Get or initialize the sentiment analysis pipeline"""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        print("Loading DistilBERT sentiment analysis model...")
        print(f"NumPy version: {np.__version__}")
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("Model loaded successfully!")
    return sentiment_pipeline

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of the input text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        # Run sentiment analysis
        pipeline = get_sentiment_pipeline()
        result = pipeline(text)[0]
        
        # Extract label and score
        label = result['label']
        score = result['score']
        
        # Map LABEL_0/LABEL_1 or POSITIVE/NEGATIVE to our format
        if label.upper() == 'POSITIVE' or label == 'LABEL_1':
            sentiment = 'positive'
        elif label.upper() == 'NEGATIVE' or label == 'LABEL_0':
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Convert score to percentage
        confidence = round(score * 100, 2)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

