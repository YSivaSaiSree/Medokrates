from flask import Flask, render_template, jsonify, request
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained Random Forest model during application startup
try:
    model = joblib.load('best_random_forest_model.joblib')
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

# Sample symptoms dataset for autocomplete suggestions
symptoms_dataset = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis'
]

label_encoder = LabelEncoder()
label_encoder.fit(symptoms_dataset)

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

    # Convert symptoms into numerical format
    try:
        encoded_symptoms = label_encoder.transform(selected_symptoms)
        prediction = model.predict([encoded_symptoms])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        app.logger.error(f'Error during prediction: {e}')
        return jsonify({'error': f'Error during prediction: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
