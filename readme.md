# AI for Personalized Medicine

A comprehensive healthcare application that leverages artificial intelligence for personalized disease prediction and treatment recommendations, including both symptom-based and genetic-based disease prediction systems.

## 🌟 Features

- **Symptom-Based Disease Prediction**

  - Interactive web interface for symptom input
  - Disease prediction using Support Vector Machine (SVC)
  - Comprehensive health recommendations including:
    - Disease descriptions
    - Precautionary measures
    - Medication suggestions
    - Dietary recommendations
    - Personalized workout plans

- **Genetic Disease Prediction**
  - Classification of genes across seven major gene families
  - Analysis using Decision Tree and Random Forest classifiers
  - Specialized categorization system:
    - Medical-related (MED)
    - Molecular Genetics Label (MGL)
    - Ribosomal processes (RHB)
    - Epigenetic processes (EPD)
    - Joint Processes (JPA)

## 🛠️ Technologies Used

- **Backend Framework:** Flask
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** scikit-learn
- **Algorithms:**
  - Support Vector Machine (SVC)
  - Decision Tree Classifier
  - Random Forest Classifier

## 📊 Datasets

The project utilizes multiple datasets:

- `symptoms_df.csv` - Symptom to numerical value mapping
- `precautions_df.csv` - Disease precautions
- `workout_df.csv` - Disease-specific workout recommendations
- `description.csv` - Disease descriptions
- `medications.csv` - Disease-specific medications
- `diets.csv` - Disease-specific dietary recommendations
- Genetic sequence dataset for gene family classification

## 🚀 Installation

1. Clone the repository

```bash
git clone [https://github.com/ADVAYA1/Disease-prediction]
```
