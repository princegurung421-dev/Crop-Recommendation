
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys

def train():
    print("Loading data...")
    try:
        crop = pd.read_csv("Crop_recommendation.csv")
    except FileNotFoundError:
        print("Error: 'Crop_recommendation.csv' not found.")
        sys.exit(1)

    print("Preprocessing data...")
    # Mapping dictionary
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
        'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
        'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
        'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }

    # Map labels to numbers
    crop['crop_num'] = crop['label'].map(crop_dict)

    # Features and Target
    X = crop.drop(['crop_num', 'label'], axis=1)
    y = crop['crop_num']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    print("Scaling features...")
    ms = MinMaxScaler()
    X_train = ms.fit_transform(X_train)
    X_test = ms.transform(X_test)

    # Model Training
    print("Training Random Forest Classifier...")
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # Evaluation
    accuracy = rfc.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Saving Model and Scaler
    print("Saving model and scaler...")
    pickle.dump(rfc, open('model.pkl', 'wb'))
    pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
    
    # We do NOT save standardscaler as it was not used in the notebook's final model logic
    print("Done! 'model.pkl' and 'minmaxscaler.pkl' have been saved.")

if __name__ == "__main__":
    train()
