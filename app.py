from flask import Flask, render_template, request, jsonify, url_for, make_response
from ml_model2 import extract_features, predict_phishing
import logging

app = Flask(__name__)

# Dummy statistics data (replace with your actual statistics)
model_statistics = {
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.88,
    'f1_score': 0.85,
    # Add more statistics as needed
}

logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.form
        domain = data.get('domain')
        
        logging.debug(f"Received domain: {domain}")
        
        features_df = extract_features(domain)
        result = predict_phishing(features_df)
        return jsonify({'result': result})
    else:
        response = make_response(render_template('index.html', statistics=model_statistics))
        response.set_cookie('session_id', '123', httponly=True)
        return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        domain = data.get('domain')
        
        logging.debug(f"Received domain: {domain}")
        
        features_df = extract_features(domain)
        result = predict_phishing(features_df)
        return jsonify({'result': result})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def statistics():
    return render_template('statistics.html', statistics=model_statistics)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
