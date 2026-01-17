"""
ASL Sign Language Recognition - Step 2: Model Training with SVM
Optimized for ASL Alphabet Dataset
Author: Your Name
Date: November 27, 2025
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# ============= CONFIGURATION =============
INPUT_FILE = "output/asl_features.csv"
MODEL_FILE = "output/asl_model_svm.pkl"
SCALER_FILE = "output/asl_scaler_svm.pkl"
CONFUSION_MATRIX_FILE = "output/confusion_matrix_svm.png"
CLASSIFICATION_REPORT_FILE = "output/classification_report_svm.txt"

# SVM hyperparameters
KERNEL = 'rbf'          # Radial Basis Function kernel (best for non-linear data)
C = 10.0                # Regularization parameter (higher = less regularization)
GAMMA = 'scale'         # Kernel coefficient ('scale' or 'auto')
RANDOM_STATE = 42       # For reproducibility
TEST_SIZE = 0.2         # 80% train, 20% test

# Visualization settings
ENABLE_VISUALIZATION = True  # Set to False to skip confusion matrix PNG (faster)

# ============= HELPER FUNCTIONS =============
def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    sys.stdout.flush()  # Force output to display immediately

def print_progress(message):
    """Print progress message and flush immediately"""
    print(message)
    sys.stdout.flush()

def save_classification_report(y_test, y_pred, labels, filename):
    """Save classification report to file"""
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    with open(filename, 'w') as f:
        f.write("ASL Sign Language Recognition - SVM Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"✅ Classification report saved: {filename}")

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    print_section("ASL Sign Language Recognition - SVM Model Training")
    
    # ============= LOAD DATA =============
    print(f"\nLoading extracted features from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ ERROR: {INPUT_FILE} not found!")
        print("   Please run 1_extract_features.py first")
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
    
    # ============= FEATURE SCALING (IMPORTANT FOR SVM!) =============
    print_section("Feature Scaling (StandardScaler)")
    
    print("⚠️  SVM requires feature scaling for optimal performance")
    print("Scaling features to zero mean and unit variance...\n")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled successfully")
    print(f"   Mean: {X_train_scaled.mean():.6f}")
    print(f"   Std: {X_train_scaled.std():.6f}")
    
    # ============= TRAIN SVM MODEL =============
    print_section("Training SVM Classifier")
    
    print(f"Hyperparameters:")
    print(f"   kernel: {KERNEL}")
    print(f"   C (regularization): {C}")
    print(f"   gamma: {GAMMA}")
    print(f"   random_state: {RANDOM_STATE}")
    print()
    
    print("⚠️  Training SVM on 109K samples may take 5-10 minutes...")
    print("Training in progress...\n")
    
    model = SVC(
        kernel=KERNEL,
        C=C,
        gamma=GAMMA,
        random_state=RANDOM_STATE,
        probability=True,  # Enable probability estimates for predict_proba
        cache_size=500,    # Increase cache for faster training
        verbose=True       # Show training progress
    )
    
    import time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✅ Training complete in {training_time:.1f} seconds ({training_time/60:.1f} minutes)!")
    
    # ============= SAVE MODEL IMMEDIATELY =============
    print_section("Saving Model Files")
    
    # Save the model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
    print(f"✅ Model saved: {MODEL_FILE} ({model_size:.2f} MB)")
    
    # Save the scaler
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    scaler_size = os.path.getsize(SCALER_FILE) / 1024
    print(f"✅ Scaler saved: {SCALER_FILE} ({scaler_size:.2f} KB)")
    
    print(f"\n💾 Total model size: {model_size:.2f} MB + {scaler_size:.2f} KB")
    
    # ============= EVALUATE MODEL =============
    print_section("Model Performance Evaluation")
    
    # Training accuracy
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Testing accuracy
    y_test_pred = model.predict(X_test_scaled)
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
    
    # ============= CONFUSION MATRIX =============
    print_section("Confusion Matrix Visualization")
    
    print_progress("Computing confusion matrix...")
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    
    try:
        print_progress("Creating visualization...")
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
        
        plt.title(f'SVM Confusion Matrix - Test Accuracy: {test_accuracy*100:.1f}%', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        print_progress("Saving confusion matrix...")
        plt.savefig(CONFUSION_MATRIX_FILE, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {CONFUSION_MATRIX_FILE}")
        
        plt.close('all')  # Close all figures to free memory
    except Exception as e:
        print(f"⚠️  Warning: Could not create visualization: {e}")
    
    sys.stdout.flush()
    
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
    
    # SVM-specific stats
    print(f"\n📊 SVM Model Info:")
    print(f"   Support vectors: {model.n_support_.sum()}")
    print(f"   Classes: {len(model.classes_)}")
    
    # ============= FINAL SUMMARY =============
    print_section("Training Complete - Summary")
    
    print(f"""
✅ SVM Model Training Successful!

📊 Performance Metrics:
   - Test Accuracy: {test_accuracy*100:.2f}%
   - Training Time: {training_time:.1f}s ({training_time/60:.1f} min)
   - Training Samples: {len(X_train)}
   - Testing Samples: {len(X_test)}
   - Classes: {len(labels)}
   - Support Vectors: {model.n_support_.sum()}
   
💾 Generated Files:
   - Model: {MODEL_FILE} ({model_size:.2f} MB)
   - Scaler: {SCALER_FILE} ({scaler_size:.2f} KB)
   - Confusion Matrix: {CONFUSION_MATRIX_FILE}
   - Classification Report: {CLASSIFICATION_REPORT_FILE}

🚀 Next Steps:
   1. Review confusion matrix for misclassifications
   2. Check classification report for detailed metrics
   3. Compare with Random Forest performance
   4. Update app_realtime.py to use SVM model (need scaler!)
    """)
    
    print("="*60)
