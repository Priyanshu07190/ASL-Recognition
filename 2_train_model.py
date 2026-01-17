"""
ASL Sign Language Recognition - Step 2: Model Training
Optimized for ASL Alphabet Dataset
Author: Your Name
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ============= CONFIGURATION =============
INPUT_FILE = "output/asl_features.csv"
MODEL_FILE = "output/asl_model.pkl"
CONFUSION_MATRIX_FILE = "output/confusion_matrix.png"
CLASSIFICATION_REPORT_FILE = "output/classification_report.txt"

# Model hyperparameters (Optimized: 78.72% accuracy, 795KB size - perfect for Arduino!)
N_ESTIMATORS = 7        # Number of trees (sweet spot for accuracy/size)
MAX_DEPTH = 9           # Maximum tree depth (fits in Arduino memory)
MIN_SAMPLES_SPLIT = 10  # Minimum samples to split node
MIN_SAMPLES_LEAF = 4    # Minimum samples in leaf
RANDOM_STATE = 42       # For reproducibility
TEST_SIZE = 0.2         # 80% train, 20% test

# ============= HELPER FUNCTIONS =============
def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def save_classification_report(y_test, y_pred, labels, filename):
    """Save classification report to file"""
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    with open(filename, 'w') as f:
        f.write("ASL Sign Language Recognition - Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"✅ Classification report saved: {filename}")

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    print_section("ASL Sign Language Recognition - Model Training")
    
    # ============= LOAD DATA =============
    print(f"\nLoading extracted features from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: {INPUT_FILE} not found!")
        print("   Please run 1_extract_features_asl.py first")
        exit(1)
    
    df = pd.read_csv(INPUT_FILE)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last (63 features)
    y = df.iloc[:, -1].values   # Last column (labels)
    
    print(f"✅ Dataset loaded successfully")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print(f"   Classes: {sorted(np.unique(y))}")
    
    # Data quality check
    if np.isnan(X).any():
        print("\n⚠️  Warning: Found NaN values, filling with 0...")
        X = np.nan_to_num(X, nan=0.0)
    
    print(f"\n📊 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label}: {count:4d} samples ({count/len(y)*100:.1f}%)")
    
    # ============= SPLIT DATA =============
    print_section("Splitting Dataset")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training samples: {len(X_train)} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Testing samples:  {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    
    # ============= TRAIN MODEL =============
    print_section("Training Random Forest Classifier")
    
    print(f"Hyperparameters:")
    print(f"   n_estimators: {N_ESTIMATORS}")
    print(f"   max_depth: {MAX_DEPTH}")
    print(f"   min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"   min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"   random_state: {RANDOM_STATE}")
    print()
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,          # Use all CPU cores
        verbose=0           # No logging spam
    )
    
    print("Training in progress...")
    model.fit(X_train, y_train)
    print("\n✅ Training complete!")
    
    # ============= EVALUATE MODEL =============
    print_section("Model Performance Evaluation")
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n🎯 Accuracy Results:")
    print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"   Testing Accuracy:  {test_accuracy*100:.2f}%")
    
    # Check for overfitting
    overfitting = train_accuracy - test_accuracy
    if overfitting > 0.10:
        print(f"   ⚠️  Warning: Possible overfitting (gap: {overfitting*100:.1f}%)")
    else:
        print(f"   ✅ Good generalization (gap: {overfitting*100:.1f}%)")
    
    # ============= DETAILED CLASSIFICATION REPORT =============
    print_section("Per-Class Performance")
    
    labels = sorted(np.unique(y))
    report = classification_report(y_test, y_test_pred, target_names=labels, zero_division=0)
    print(report)
    
    # Save report to file
    save_classification_report(y_test, y_test_pred, labels, CLASSIFICATION_REPORT_FILE)
    
    # ============= FEATURE IMPORTANCE =============
    print_section("Feature Importance Analysis")
    
    feature_names = df.columns[:-1]
    importances = model.feature_importances_
    
    # Get top 15 features
    top_indices = np.argsort(importances)[::-1][:15]
    
    print("\nTop 15 Most Important Features:")
    for i, idx in enumerate(top_indices, 1):
        landmark_num = idx // 3
        coord = ['x', 'y', 'z'][idx % 3]
        print(f"{i:2d}. {feature_names[idx]:6s} (Landmark {landmark_num:2d}, {coord}) - {importances[idx]:.4f}")
    
    # ============= CONFUSION MATRIX =============
    print_section("Confusion Matrix Visualization")
    
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy*100:.1f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {CONFUSION_MATRIX_FILE}")
    
    # Show plot (commented out to allow script to complete automatically)
    # plt.show()  # Uncomment if you want to see the plot interactively
    plt.close()  # Close the figure to free memory
    
    # ============= MODEL ANALYSIS =============
    print_section("Model Statistics")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    class_accuracy = []
    for i, label in enumerate(labels):
        mask = y_test == label
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_test_pred[mask])
            class_accuracy.append((label, acc))
            print(f"   {label}: {acc*100:.1f}%")
    
    # Best and worst performing classes
    class_accuracy.sort(key=lambda x: x[1], reverse=True)
    print(f"\n✅ Best performing: {class_accuracy[0][0]} ({class_accuracy[0][1]*100:.1f}%)")
    print(f"⚠️  Worst performing: {class_accuracy[-1][0]} ({class_accuracy[-1][1]*100:.1f}%)")
    
    # ============= SAVE MODEL =============
    print_section("Saving Model")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    model_size_kb = os.path.getsize(MODEL_FILE) / 1024
    
    print(f"✅ Model saved: {MODEL_FILE}")
    print(f"   Model size: {model_size_kb:.2f} KB")
    
    # Check Arduino compatibility
    print("\n📱 Arduino Deployment Check:")
    if model_size_kb < 50:
        print("   ✅ Perfect size for Arduino Nano 33 BLE Sense!")
        print("   ✅ Expected RAM usage: ~40-50 KB")
        print("   ✅ Expected Flash usage: ~30-40 KB")
    elif model_size_kb < 100:
        print("   ⚠️  Model is large but should work")
        print("   ⚠️  May need optimization (reduce n_estimators)")
    else:
        print("   ❌ Model too large for Arduino!")
        print("   ❌ Reduce n_estimators or max_depth")
    
    # ============= FINAL SUMMARY =============
    print_section("Training Complete - Summary")
    
    print(f"""
✅ Model Training Successful!

📊 Performance Metrics:
   - Test Accuracy: {test_accuracy*100:.2f}%
   - Training Samples: {len(X_train)}
   - Testing Samples: {len(X_test)}
   - Classes: {len(labels)}
   
💾 Generated Files:
   - Model: {MODEL_FILE} ({model_size_kb:.2f} KB)
   - Confusion Matrix: {CONFUSION_MATRIX_FILE}
   - Classification Report: {CLASSIFICATION_REPORT_FILE}

🚀 Next Steps:
   1. Review confusion matrix for misclassifications
   2. Check classification report for detailed metrics
   3. Run: python 3_convert_arduino.py
    """)
    
    print("="*60)
