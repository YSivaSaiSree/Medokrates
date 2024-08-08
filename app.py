from flask import Flask, render_template, jsonify, request
import joblib

app = Flask(__name__)

# Load the trained Random Forest model during application startup
try:
    model = joblib.load('best_random_forest_model.joblib')
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

# Sample symptoms dataset for autocomplete suggestions
symptoms_dataset = ['itching', 'skin_rash', 'nodal_skin_eruptions', ...]  # Your complete list

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_dataset)

@app.route('/suggest-symptoms')
def suggest_symptoms():
    partial_input = request.args.get('partialInput', '')
    suggestions = [symptom for symptom in symptoms_dataset if partial_input.lower() in symptom.lower()]
    return jsonify(suggestions)

@app.route('/classify-disease', methods=['POST'])
def classify_disease_endpoint():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    selected_symptoms = request.json.get('selectedSymptoms', [])
    try:
        prediction = model.predict([selected_symptoms])[0]  # Assuming your model takes a list of symptoms
        return jsonify({'prediction': prediction})
    except Exception as e:
        app.logger.error(f'Error during prediction: {e}')
        return jsonify({'error': f'Error during prediction: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
