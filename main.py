import os
import sys
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from flask import Flask, request, render_template

# Print current working directory and Python executable
print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed working directory to:", os.getcwd())

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Flask app
app = Flask(__name__)


# Load database datasets
datasets = {
    'sym_des': 'symtoms_df.csv',
    'precautions': 'precautions_df.csv',
    'workout': 'workout_df.csv',
    'description': 'description.csv',
    'medications': 'medications.csv',
    'diets': 'diets.csv'
}

for var, filename in datasets.items():
    filepath = os.path.join('datasets', filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)
    globals()[var] = pd.read_csv(filepath)
    print(f"Loaded {filename}")

# Load the trained pipeline
model_path = os.path.join('models', 'svc_pipeline.pkl')
print("Attempting to open:", os.path.abspath(model_path))

if not os.path.exists(model_path):
    print(f"Error: The model file {model_path} does not exist.")
    print("Please run train_model.py to generate the model file.")
    sys.exit(1)

try:
    with open(model_path, 'rb') as model_file:
        pipeline = pickle.load(model_file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# Helper function to retrieve disease information
def helper(dis):
    desc =['Disease'] == dis['Description']
    desc = " ".join([w for w in desc])

    pre = ['Disease'] == dis[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values[0] if pd.notna(col)]

    med = ['Disease'] == dis['Medication']
    med = [med for med in med.values if pd.notna(med)]

    die = ['Disease'] == dis['Diet']
    die = [die for die in die.values if pd.notna(die)]

    wrkout = ['disease'] == dis['workout'].values[0]

    return desc, pre, med, die, wrkout

# Define the symptom and disease dictionaries
symptoms_dict = {symptom: index for index, symptom in enumerate(sym_des.columns[1:])}
diseases_list = sym_des['Disease'].unique().tolist()

# Create a mapping of symptom variations to standard symptom names
symptom_variations = {
    'itch': 'itching',
    'itchiness': 'itching',
    'rash': 'skin_rash',
    'sneezing': 'continuous_sneezing',
    'shivers': 'shivering',
    'stomach ache': 'stomach_pain',
    'throw up': 'vomiting',
    'threw up': 'vomiting',
    'throwing up': 'vomiting',
    'puke': 'vomiting',
    'puking': 'vomiting',
    'tired': 'fatigue',
    'exhausted': 'fatigue',
    'worry': 'anxiety',
    'worried': 'anxiety',
    'nervousness': 'anxiety',
    'sugar': 'irregular_sugar_level',
    'coughing': 'cough',
    'fever': 'high_fever',
    'high temperature': 'high_fever',
    'shortness of breath': 'breathlessness',
    'short of breath': 'breathlessness',
    'cant breathe': 'breathlessness',
    'sweaty': 'sweating',
    'thirsty': 'dehydration',
    'headaches': 'headache',
    'migraine': 'headache',
    'yellow skin': 'yellowish_skin',
    'yellow eyes': 'yellowing_of_eyes',
    'nauseous': 'nausea',
    'no appetite': 'loss_of_appetite',
    'eye pain': 'pain_behind_the_eyes',
    'backache': 'back_pain',
    'stomach ache': 'abdominal_pain',
    'diarrhea': 'diarrhoea',
    'runny poop': 'diarrhoea',
    'watery stool': 'diarrhoea',
}

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize_and_match_symptoms(user_input):
    # Tokenize the input
    tokens = word_tokenize(user_input.lower())
    
    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Match tokens to standard symptom names
    matched_symptoms = []
    for token in lemmatized_tokens:
        if token in symptoms_dict:
            matched_symptoms.append(token)
        elif token in symptom_variations:
            matched_symptoms.append(symptom_variations[token])
    
    return list(set(matched_symptoms))  # Remove duplicates

# Model Prediction function
def get_top_predicted_values(patient_symptoms):
    # Create a DataFrame with all symptoms set to 0
    input_df = pd.DataFrame([[0] * len(symptoms_dict)], columns=symptoms_dict.keys())
    
    # Set the patient's symptoms to 1
    for symptom in patient_symptoms:
        if symptom in input_df.columns:
            input_df[symptom] = 1
    
    # Get the predicted probabilities for all diseases
    predicted_probabilities = pipeline.predict_proba(input_df)[0]
    
    # Get the indices of the top 3 predictions
    top_indices = np.argsort(predicted_probabilities)[-3:][::-1]
    
    # Prepare a list of top 3 diseases and their probabilities
    top_diseases = [(diseases_list[idx], round(predicted_probabilities[idx] * 100, 2)) for idx in top_indices]
    
    return top_diseases  # Return the list of top 3 diseases and their probabilities

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(f"Received symptoms: {symptoms}")
        if not symptoms or symptoms.lower() == "symptoms":
            message = "Please enter your symptoms."
            return render_template('index.html', message=message)
        else:
            # Tokenize and match user input to standard symptom names
            matched_symptoms = tokenize_and_match_symptoms(symptoms)
            print(f"Matched symptoms: {matched_symptoms}")
            
            if not matched_symptoms:
                message = "No matching symptoms found. Please check your input and try again."
                return render_template('index.html', message=message)
            
            predicted_diseases = get_top_predicted_values(matched_symptoms)  # Get top 3 diseases
            print(f"Predicted diseases: {predicted_diseases}")

            disease_info = []
            for disease, prob in predicted_diseases:
                desc, pre, med, rec_diet, wrkout = helper(disease)
                disease_info.append({
                    'name': disease,
                    'probability': prob,
                    'description': desc,
                    'precautions': pre,
                    'medications': med,
                    'diet': rec_diet,
                    'workout': wrkout
                })

            return render_template('index.html', predicted_diseases=disease_info, input_symptoms=', '.join(matched_symptoms))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)