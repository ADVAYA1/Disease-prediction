import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Print current working directory and Python executable
print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Changed working directory to:", os.getcwd())

# Load your actual dataset
dataset_path = os.path.join('datasets', 'symtoms_df.csv')
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found.")
    sys.exit(1)

sym_des = pd.read_csv(dataset_path)
print(f"Loaded dataset from {dataset_path}")

# Prepare your features (X) and target (y)
X = sym_des.drop('Disease', axis=1)  # Assuming 'Disease' is your target column
y = sym_des['Disease']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create a ColumnTransformer for one-hot encoding
# Create a ColumnTransformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), X.columns)
    ])

# Create a pipeline with the preprocessor and SVC
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
# Fit the pipeline
pipeline.fit(X_train, y_train)

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained pipeline to a file
model_path = os.path.join('models', 'svc_pipeline.pkl')
print(f"Saving model to {os.path.abspath(model_path)}...")

try:
    with open(model_path, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {str(e)}")
    sys.exit(1)

print(f"Number of features after one-hot encoding: {pipeline.named_steps['preprocessor'].transform(X).shape[1]}")

# Optional: Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Model training and saving completed successfully.")