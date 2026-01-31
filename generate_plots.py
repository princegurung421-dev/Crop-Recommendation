
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Set style
sns.set_style("whitegrid")

# Create plots directory if it doesn't exist
if not os.path.exists('static/images/plots'):
    os.makedirs('static/images/plots')

# Load data
try:
    print("Loading data...")
    df = pd.read_csv('Crop_recommendation.csv')
    
    # --- Data Preprocessing for Model Plots ---
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
        'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
        'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
        'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }
    df['crop_num'] = df['label'].map(crop_dict)
    X = df.drop(['crop_num', 'label'], axis=1)
    y = df['crop_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ms = MinMaxScaler()
    X_train_scaled = ms.fit_transform(X_train)
    X_test_scaled = ms.transform(X_test)
    
    print("Training temporary model for plots...")
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)
    y_pred = rfc.predict(X_test_scaled)
    
    # --- GENERATING PLOTS ---

    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=crop_dict.keys(), yticklabels=crop_dict.keys())
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/plots/confusion_matrix.png')
    plt.close()

    # 2. Feature Importance
    print("Generating Feature Importance...")
    plt.figure(figsize=(10, 6))
    importances = rfc.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]
    
    sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/images/plots/feature_importance.png')
    plt.close()

    # 3. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop('crop_num', axis=1) # Exclude label encoding
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation between Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('static/images/plots/correlation_heatmap.png')
    plt.close()

    # 4. Class Distribution
    print("Generating Class Distribution...")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=df, palette='Spectral')
    plt.xticks(rotation=90)
    plt.title('Distribution of Crops in Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('static/images/plots/class_distribution.png')
    plt.close()

    print("All plots generated successfully!")

except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc()
