"""
Train SVM model for ASL phrase recognition
Uses features extracted from augmented phrase images
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

INPUT_FILE = "output/asl_phrase_features.csv"
MODEL_FILE = "output/asl_phrase_model.pkl"
SCALER_FILE = "output/asl_phrase_scaler.pkl"

def train_phrase_model():
    print("="*60)
    print("ASL Phrase Recognition - Model Training")
    print("="*60)
    print()
    
    # Load data
    print("📂 Loading features...")
    df = pd.read_csv(INPUT_FILE)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Phrases: {len(np.unique(y))}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Scale features
    print("⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    print("🎯 Training SVM model (this may take a few minutes)...")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print()
    print("="*60)
    print(f"✅ Training Complete!")
    print("="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print()
    
    # Classification report
    print("📋 Classification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # Save report
    with open('output/phrase_classification_report.txt', 'w') as f:
        f.write(f"ASL Phrase Recognition Model\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    
    # Confusion matrix
    print("📈 Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y),
                cbar_kws={'label': 'Count'})
    plt.title(f'ASL Phrase Recognition - Confusion Matrix\\nAccuracy: {accuracy*100:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Phrase', fontsize=12)
    plt.xlabel('Predicted Phrase', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('output/phrase_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: output/phrase_confusion_matrix.png")
    
    # Save model and scaler
    print()
    print("💾 Saving model...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved: {MODEL_FILE}")
    
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: {SCALER_FILE}")
    
    # Show per-class accuracy
    print()
    print("📊 Per-phrase accuracy (top 10 best):")
    phrase_accuracy = {}
    for phrase in np.unique(y):
        mask = y_test == phrase
        if mask.sum() > 0:
            phrase_acc = accuracy_score(y_test[mask], y_pred[mask])
            phrase_accuracy[phrase] = phrase_acc
    
    sorted_phrases = sorted(phrase_accuracy.items(), key=lambda x: x[1], reverse=True)
    for phrase, acc in sorted_phrases[:10]:
        print(f"   {phrase:20s}: {acc*100:.1f}%")
    
    print()
    print("📊 Per-phrase accuracy (bottom 10 worst):")
    for phrase, acc in sorted_phrases[-10:]:
        print(f"   {phrase:20s}: {acc*100:.1f}%")
    
    print()
    print("✅ All done! Run app_phrase_recognition.py to test live")

if __name__ == "__main__":
    train_phrase_model()
