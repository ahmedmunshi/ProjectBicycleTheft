# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os


def train_model():
    # Get the current directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)  # Go up one level to project root

    # 1. Load the dataset
    data_path = '/Users/ahmedmunshi/Downloads/ProjectBikeTheftCOMP309/Bicycle_Thefts_Open_Data.csv'
    print(f"Looking for data file at: {data_path}")
    data = pd.read_csv(data_path)

    print("Data loaded successfully!")

    # 2. Prepare data for modeling
    features = ['OCC_YEAR', 'OCC_HOUR', 'BIKE_COST']
    target = 'STATUS'

    # Handle missing values in BIKE_COST
    data['BIKE_COST'] = data['BIKE_COST'].fillna(data['BIKE_COST'].mean())

    # Prepare X (features) and y (target)
    X = data[features]
    y = data[target]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Apply SMOTE to balance the classes
    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    # Train model with balanced data
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model and scaler with absolute paths
    model_path = os.path.join(project_dir, 'model.pkl')
    scaler_path = os.path.join(project_dir, 'scaler.pkl')

    print(f"Saving model to: {model_path}")
    print(f"Saving scaler to: {scaler_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("\nModel and scaler saved successfully!")

    return model, scaler


if __name__ == "__main__":
    train_model()