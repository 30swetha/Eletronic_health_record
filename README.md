# Eletronic_health_record
EHR Patient Outcome Prediction (Text + Tabular Features)
This project demonstrates a multi-modal machine learning approach to predict patient outcomes (specifically, hospital readmission) using a combination of clinical notes (text data) and structured tabular features (age, gender, prior admissions, etc.). The solution is built with TensorFlow 2.x and Keras, suitable for running in Google Colab or a local environment.

Table of Contents
Project Overview
Dataset
Model Architecture
Setup and Installation
How to Run the Code
Model Evaluation
Example Inference
Saved Artifacts
Project Overview
Predicting patient outcomes is crucial in healthcare for improving care quality and resource allocation. This project tackles the problem of readmission prediction by leveraging both the rich contextual information from clinical notes and the structured demographic/clinical data. It employs a deep learning model that processes text data using an Embedding layer and Bidirectional LSTMs, and tabular data using Dense layers, before combining these features for a final prediction.

Dataset
The project uses a synthetic dataset generated to mimic real-world Electronic Health Record (EHR) data. Each patient record consists of:

note: A synthetic clinical note (text data) with keywords correlated to the readmission outcome.
age: Patient's age.
gender: Patient's gender (0 for female, 1 for male).
num_prior: Number of prior admissions.
los: Length of stay in days.
lab_score: A synthetic laboratory score.
readmit: The binary target variable (0 for no readmission, 1 for readmission).
The dataset is split into training, validation, and test sets, with stratification to maintain class balance. Tabular features are standardized using StandardScaler.

Model Architecture
The model is a multi-input Keras model comprising two main branches:

Text Branch:

Takes raw clinical notes as input.
Uses TextVectorization for tokenization and sequence padding.
An Embedding layer converts token IDs into dense vectors.
A Bidirectional LSTM layer captures sequential dependencies in the text.
GlobalMaxPool1D reduces the sequence to a fixed-size vector.
Followed by Dense layers and Dropout for regularization.
Tabular Branch:

Takes standardized tabular features as input.
Consists of Dense layers, BatchNormalization, and Dropout for processing structured data.
These two branches are concatenated, and the combined features are passed through additional Dense layers to produce a final binary classification output (probability of readmission) via a sigmoid activation function.

The model is compiled with Adam optimizer, binary_crossentropy loss, and BinaryAccuracy and AUC metrics.

Setup and Installation
To run this project, you'll need Python 3.x and the following libraries. If running in Google Colab, most are pre-installed.

pip install numpy pandas tensorflow scikit-learn
How to Run the Code
The provided notebook (EHR_Patient_Outcome_Prediction.ipynb) is structured sequentially. You can run each cell in order:

Imports and Seed Setting: Initializes necessary libraries and sets random seeds for reproducibility.
Synthetic Dataset Generation: Creates the synthetic patient data.
Train/Val/Test Split: Divides the dataset and scales tabular features.
Text Preprocessing: Adapts TextVectorization on the training text data.
Build TF Dataset pipelines: Creates tf.data.Dataset objects for efficient training.
Calculate Class Weights: Computes class weights to handle potential class imbalance.
Model Definition and Compilation: Defines the multi-modal Keras model.
Callbacks: Sets up ReduceLROnPlateau, EarlyStopping, and ModelCheckpoint for robust training.
Training: Trains the model on the generated datasets.
Evaluation: Evaluates the model's performance on the test set and prints classification metrics.
Example Inference: Demonstrates how to use the trained model for single-patient predictions.
Save Preprocessing Objects and Model: Saves the trained model, TextVectorization vocabulary, and StandardScaler parameters for future use.
Model Evaluation
After training, the model is evaluated on the held-out test set. Key metrics reported include:

Loss: Binary cross-entropy loss.
Accuracy: Overall prediction accuracy.
AUC (Area Under the Receiver Operating Characteristic Curve): A measure of the model's ability to distinguish between positive and negative classes.
Classification Report: Provides precision, recall, and F1-score for each class.
Confusion Matrix: Shows the counts of true positives, true negatives, false positives, and false negatives.
Example Inference
The predict_single function demonstrates how to make a prediction for a new patient. It takes a clinical note (string) and tabular features (array) as input, preprocesses them using the saved text_vectorizer and scaler, and then uses the model to output a readmission probability.

example_note = "Patient with fever and cough, worsening shortness of breath. Recurrent admissions for pneumonia."
example_tab = [72, 1, 2, 6, 1.5]  # age, gender, num_prior, los, lab_score
probability = predict_single(example_note, example_tab)
print(f"Predicted probability of readmission: {probability}")
Saved Artifacts
After execution, the following files are saved:

ehr_text_tab_model.json: Keras model architecture in JSON format.
ehr_text_tab_model_weights.weights.h5: Model weights.
text_vectorizer_vocab.txt: Vocabulary used by the TextVectorization layer.
tabular_scaler_params.npz: Mean and scale parameters of the StandardScaler.
