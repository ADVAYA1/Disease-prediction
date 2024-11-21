from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
from collections import Counter
import re
from itertools import product
import json
import ast

# flask app
app = Flask(__name__)

GEMINI_API_KEY="AIzaSyDIPN5giDG4kQujeCqpvkGgZHo1PcdYRP8"
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)


# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

with open('Datasets/family_des.json') as json_file:
    genetic_info = json.load(json_file)

# load model===========================================
svc = pickle.load(open('models/svc_prob.pkl','rb'))

class GeneticSequenceClassifier:
    def __init__(self, k=4):
        self.k = k
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
    
    def get_sequence_features(self, sequence):
        # Calculate additional sequence features
        length = len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        
        # Calculate nucleotide frequencies
        nucleotides = ['A', 'T', 'G', 'C']
        nucleotide_freqs = {n: sequence.count(n)/length for n in nucleotides}
        
        # Calculate dinucleotide frequencies
        dinucleotides = [''.join(p) for p in product(nucleotides, repeat=2)]
        dinuc_freqs = {}
        for di in dinucleotides:
            count = len(re.findall(f'(?={di})', sequence))
            dinuc_freqs[di] = count/(length-1) if length > 1 else 0
        
        # Calculate positional features
        start_tri = sequence[:3] if len(sequence) >= 3 else sequence.ljust(3, 'N')
        end_tri = sequence[-3:] if len(sequence) >= 3 else sequence.rjust(3, 'N')
        
        return {
            'length': length,
            'gc_content': gc_content,
            **nucleotide_freqs,
            **dinuc_freqs,
            'start': start_tri,
            'end': end_tri
        }
    
    def extract_features(self, sequences):
        print("Extracting enhanced features...")
        
        # Get all sequence features
        seq_features = []
        for seq in tqdm(sequences, desc="Extracting sequence features"):
            seq_features.append(self.get_sequence_features(seq))
        
        # Convert sequence features to DataFrame
        features_df = pd.DataFrame(seq_features)
        
        # One-hot encode start and end sequences
        start_encoded = pd.get_dummies(features_df['start'], prefix='start')
        end_encoded = pd.get_dummies(features_df['end'], prefix='end')
        features_df = pd.concat([features_df.drop(['start', 'end'], axis=1), 
                                  start_encoded, end_encoded], axis=1)
        
        # Extract k-mer frequencies
        all_kmers = set()
        kmer_counts = []
        
        # First pass - collect unique k-mers
        for seq in tqdm(sequences, desc="Collecting k-mers"):
            for i in range(len(seq) - self.k + 1):
                all_kmers.add(seq[i:i + self.k])
        
        # Second pass - count k-mers
        for seq in tqdm(sequences, desc="Counting k-mers"):
            counts = Counter()
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                counts[kmer] += 1
            kmer_counts.append(counts)
        
        # Create k-mer frequency matrix
        kmer_df = pd.DataFrame(kmer_counts).fillna(0)
        
        # Combine all features
        final_features = pd.concat([features_df, kmer_df], axis=1)
        return final_features
    
    def fit(self, X, y):
        print("Starting feature extraction for training data...")
        X_features = self.extract_features(X)
        self.feature_names = X_features.columns
        
        # Scale features
        X_features = self.scaler.fit_transform(X_features)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Initialize model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=50,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        print("Training RandomForest...")
        self.model.fit(X_features, y_encoded)
        return self
    
    def predict(self, X):
        X_features = self.extract_features(X)
        
        # Ensure features match training features
        missing_cols = set(self.feature_names) - set(X_features.columns)

        # Create a DataFrame for missing columns filled with 0s
        if missing_cols:
            missing_df = pd.DataFrame(0, index=np.arange(len(X_features)), columns=list(missing_cols))  # Convert to list
            X_features = pd.concat([X_features, missing_df], axis=1)

        # Reorder the DataFrame to match the training feature names
        X_features = X_features[self.feature_names]

        # Scale features
        X_features = self.scaler.transform(X_features)

        y_pred = self.model.predict(X_features)
        return self.label_encoder.inverse_transform(y_pred)


    def save(self, filename):
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'k': self.k,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
    
    @classmethod
    def load(cls, filename):
        model_data = joblib.load(filename)
        classifier = cls(k=model_data['k'])
        classifier.model = model_data['model']
        classifier.label_encoder = model_data['label_encoder']
        classifier.scaler = model_data['scaler']
        classifier.feature_names = model_data['feature_names']
        return classifier

# Load and prepare the data
print("Loading data...")
genetic_data = pd.read_csv('Datasets/genetic.csv')
family_data = pd.read_csv('Datasets/family.txt')

# Map class labels to gene families
family_mapping = {
    row['Class label']: row['Gene family'] 
    for _, row in family_data.iterrows()
}
genetic_data['class'] = genetic_data['class_label'].map(family_mapping)

# Split data with stratification to ensure balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    genetic_data['sequence'],
    genetic_data['class'],
    test_size=0.3,
    random_state=42,
    stratify=genetic_data['class']
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Train the model
print("\nTraining model...")
classifier = GeneticSequenceClassifier(k=4)
classifier.fit(X_train, y_train)

# Save the model
print("\nSaving model...")
classifier.save('genetic_classifier.pkl')

# Evaluate on test set
print("\nEvaluating model on test set...")
y_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


classifier = GeneticSequenceClassifier.load('genetic_classifier.pkl')
#============================================================
#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med_series = medications[medications['Disease'] == dis]['Medication']

    if not med_series.empty:
        med = med_series.iloc[0]  # Extract the first value as a string
    else:
        print(f"No data found for disease: {dis}")
        med = None

    # Proceed with cleaning and parsing if med is not None
    if med:
        # Clean the string to fix any formatting issues
        med = med.strip()  # Remove any leading or trailing whitespace
        med = med.replace("'", "\"")  # Replace single quotes with double quotes

        # Convert the string to a list using ast.literal_eval
        try:
            med_list = ast.literal_eval(med)
        except ValueError as e:
            print("Error parsing the medication list:", e)

    die_series = diets[diets['Disease'] == dis]['Diet']

    # Ensure you're getting the first value (string) from the Series
    if not die_series.empty:
        die = die_series.iloc[0]  # Extract the first value as a string
    else:
        print(f"No data found for disease: {dis}")
        die = None

    # Proceed with cleaning and parsing if die is not None
    if die:
        # Clean the string to fix any formatting issues
        die = die.strip()  # Remove any leading or trailing whitespace
        die = die.replace("'", "\"")  # Replace single quotes with double quotes

        # Convert the string to a list using ast.literal_eval
        try:
            die_list = ast.literal_eval(die)
        except ValueError as e:
            print("Error parsing the diet list:", e)

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med_list,die_list,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#41, 132
# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    print(input_vector.shape)
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    print(svc.predict([input_vector]))
    
    # Get the probabilities for all classes
    probabilities = svc.predict_proba([input_vector])[0]

    # Get the top 3 indices of the diseases with the highest probabilities
    top_indices = np.argsort(probabilities)[::-1][:3]

    # Get the corresponding disease names and their probabilities, converting probabilities to standard float
    top_diseases = [(diseases_list[idx], float(probabilities[idx])*100) for idx in top_indices]
    
    return top_diseases


def get_genetic_details(result):
    # Assuming result is the key that corresponds to the gene family
    details = genetic_info.get(result, {})
    return details      

@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    # Initialize results
    disease_prediction = None
    genetic_result = None
    message = ""
    
    # Initialize variables for disease prediction details
    dis_des = None
    precautions = None
    medications = None
    rec_diet = None
    workout = None
    top_diseases = None
    genetic_details={}
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        gene_sequence = request.form.get('gene_sequence', '').strip()

        context = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
        

        if symptoms:
            query = f"Use the provided {context} only to convert {symptoms} into one of the dictionary items then return that item, no explanation, just one word answer."
            response = model.generate_content(query)
            symptoms = response.text.strip()

            if symptoms == "Symptoms":
                message = "Please either write symptoms or you have written misspelled symptoms."
            else:
                # Process symptoms into a list
                user_symptoms = [s.strip() for s in symptoms.split(',')]
                user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
                top_diseases = get_predicted_value(user_symptoms)

                # Get details for the first predicted disease (handle cases if the list is empty)
                if top_diseases:
                    dis_des, precautions, medications, rec_diet, workout = helper(top_diseases[0][0])
                    my_precautions = [i for i in precautions[0]] if precautions else []
                else:
                    message = "No diseases predicted based on the provided symptoms."
        else:
            message = "Please provide symptoms."

        # Handle gene sequence input only if it exists
        if gene_sequence:
            try:
                genetic_result = classifier.predict([gene_sequence])[0]
                genetic_details = get_genetic_details(genetic_result)
                print(genetic_result, genetic_details)
            except Exception as e:
                print(f"Error in genetic prediction: {e}")
                genetic_result = None
                genetic_details={}
        else:
            genetic_result = None  # Ensure genetic_result is None if gene_sequence is not provided
            genetic_details = {}  # Initialize an empty dictionary for genetic details

        # Render results based on conditions
        return render_template('index.html',
                               message=message,
                               predicted_disease=top_diseases,
                               dis_des=dis_des,
                               my_precautions=my_precautions if 'my_precautions' in locals() else [],
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout,
                               genetic_result=genetic_result,  # Pass the genetic result
                               genetic_details=genetic_details)

    return render_template('index.html', genetic_result=None)

if __name__ == '__main__':
    app.run(debug=True)