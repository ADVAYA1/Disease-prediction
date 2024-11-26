import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Import the Support Vector Classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
from collections import Counter
import re
from itertools import product

class GeneticSequenceClassifier:
    def __init__(self, k=4, model_type='random_forest'):
        self.k = k
        self.model_type = model_type  # 'random_forest' or 'svc'
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
        
        # Initialize and train the model based on the model_type
        if self.model_type == 'random_forest':
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
        elif self.model_type == 'svc':
            self.model = SVC(kernel='linear', C=1, random_state=42)  # Using linear kernel for simplicity
            print("Training Support Vector Classifier...")
        else:
            raise ValueError(f"Model type {self.model_type} is not recognized.")
        
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

# Train and evaluate RandomForest
print("\nTraining RandomForest model...")
rf_classifier = GeneticSequenceClassifier(k=4, model_type='random_forest')
rf_classifier.fit(X_train, y_train)

# Evaluate RandomForest on test set
print("\nEvaluating RandomForest model on test set...")
y_pred_rf = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"RandomForest Test Set Accuracy: {rf_accuracy:.4f}")

# Train and evaluate Support Vector Classifier (SVC)
print("\nTraining Support Vector Classifier (SVC) model...")
svc_classifier = GeneticSequenceClassifier(k=4, model_type='svc')
svc_classifier.fit(X_train, y_train)

# Evaluate SVC on test set
print("\nEvaluating SVC model on test set...")
y_pred_svc = svc_classifier.predict(X_test)
svc_accuracy = accuracy_score(y_test, y_pred_svc)
print(f"SVC Test Set Accuracy: {svc_accuracy:.4f}")

# Compare the accuracies
print("\nAccuracy Comparison:")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVC Accuracy: {svc_accuracy:.4f}")

# Optionally, you can print the classification reports for more detailed performance comparison
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nSVC Classification Report:")
print(classification_report(y_test, y_pred_svc))

# Save both models
print("\nSaving models...")
rf_classifier.save('geneticrf.pkl')
svc_classifier.save('geneticsvc.pkl')

# Assuming the 'GeneticSequenceClassifier' class and the models are already defined and saved.

# Load the trained models
rf_classifier = GeneticSequenceClassifier.load('geneticrf.pkl')
svc_classifier = GeneticSequenceClassifier.load('geneticsvc.pkl')

# Define the gene sequence for testing
test_sequence = "AGCTC"

# Predict using the Random Forest Classifier
rf_prediction = rf_classifier.predict([test_sequence])  # Wrap in list for single sequence
print(f"Random Forest Prediction for '{test_sequence}': {rf_prediction[0]}")

# Predict using the Support Vector Classifier (SVC)
svc_prediction = svc_classifier.predict([test_sequence])  # Wrap in list for single sequence
print(f"SVC Prediction for '{test_sequence}': {svc_prediction[0]}")
