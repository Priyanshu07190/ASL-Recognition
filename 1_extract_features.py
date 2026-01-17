# 1_extract_features_asl.py - Optimized for ASL Alphabet Dataset

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

# ============= CONFIGURATION FOR ASL ALPHABET DATASET =============
DATASET_PATH = r"C:\Users\radhe\OneDrive\Desktop\ML_project\dataset\asl_alphabet_train\asl_alphabet_train"
MAX_IMAGES_PER_CLASS = None  # Process ALL images (None = unlimited)
OUTPUT_FILE = "output/asl_features.csv"

# Classes to EXCLUDE (these require motion, not static signs)
EXCLUDE_CLASSES = ['del', 'nothing', 'space']  # Optional: exclude these

# ============= INITIALIZE MEDIAPIPE =============
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Standard confidence
    min_tracking_confidence=0.5
)
print("✅ MediaPipe initialized\n")

# ============= FUNCTIONS =============
def extract_landmarks(image_path):
    """
    Extract hand landmarks from ASL Alphabet image
    These images are 200x200 color photos with good quality
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # ASL Alphabet images are already good quality
        # Just convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks
        return None
            
    except Exception as e:
        return None


def process_asl_dataset(dataset_path, max_per_class=None, exclude_classes=None):
    """
    Process ASL Alphabet Dataset structure
    """
    data = []
    labels = []
    skipped = 0
    excluded = 0
    
    # Get all class folders
    class_folders = sorted([
        f for f in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, f))
    ])
    
    # Remove excluded classes
    if exclude_classes:
        class_folders = [c for c in class_folders if c not in exclude_classes]
        print(f"Excluding classes: {exclude_classes}\n")
    
    print(f"Found {len(class_folders)} classes: {class_folders}\n")
    
    # Process each class
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        
        # ASL Alphabet uses .jpg format
        image_files = glob(os.path.join(class_path, '*.jpg'))
        
        # Limit images if specified
        if max_per_class:
            image_files = image_files[:max_per_class]
        
        print(f"Processing '{class_name}': {len(image_files)} images")
        
        # Extract features with progress bar
        for image_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            landmarks = extract_landmarks(image_path)
            
            if landmarks is not None:
                data.append(landmarks)
                labels.append(class_name)
            else:
                skipped += 1
    
    # Create DataFrame
    columns = [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Extraction Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(data)} images")
    print(f"Skipped (no hand detected): {skipped} images")
    print(f"Excluded classes: {excluded}")
    print(f"Total classes: {len(df['label'].unique())}")
    print(f"Features per image: 63")
    
    return df


# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    print("="*60)
    print("ASL Alphabet Dataset - Feature Extraction")
    print("="*60)
    print()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset path not found!")
        print(f"   Expected: {DATASET_PATH}")
        print(f"\n   Please extract asl-alphabet.zip to the correct location.")
        exit(1)
    
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Processing {MAX_IMAGES_PER_CLASS} images per class")
    print(f"This will take approximately 15-20 minutes...\n")
    
    # Process dataset
    df = process_asl_dataset(
        DATASET_PATH, 
        MAX_IMAGES_PER_CLASS,
        EXCLUDE_CLASSES
    )
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE) / 1024
    
    print(f"\n✅ Features saved to: {OUTPUT_FILE}")
    print(f"   File size: {file_size:.1f} KB")
    
    # Display class distribution
    print(f"\nSamples per class:")
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  {label}: {count:3d} samples")
    
    # Preview
    print(f"\nFirst few samples:")
    print(df.head(3))
    
    print(f"\n{'='*60}")
    print("✅ Step 1 Complete!")
    print(f"   Dataset: {len(df)} samples, {len(class_counts)} classes")
    print(f"   Next: Run 2_train_model.py")
    print(f"{'='*60}")
