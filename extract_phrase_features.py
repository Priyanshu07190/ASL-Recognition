"""
Extract MediaPipe hand landmarks from ASL phrase images
Converts augmented phrase images to feature vectors for training
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Detect both hands
    min_detection_confidence=0.5
)

INPUT_DIR = "dataset/asl_phrases_personal"
OUTPUT_FILE = "output/asl_phrase_features.csv"

def extract_landmarks_from_image(image_path):
    """Extract 126 hand landmark features from image (both hands)"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        
        # Extract landmarks for up to 2 hands
        for hand_idx in range(min(2, len(results.multi_hand_landmarks))):
            hand_landmarks = results.multi_hand_landmarks[hand_idx]
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # If only 1 hand detected, pad with zeros for second hand
        if len(results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0] * 63)  # Pad with zeros for missing hand
        
        return landmarks if len(landmarks) == 126 else None
    
    return None

def main():
    print("="*60)
    print("ASL Phrase Feature Extraction")
    print("="*60)
    print()
    
    # Get all phrase folders
    phrase_dirs = sorted([d for d in Path(INPUT_DIR).iterdir() if d.is_dir()])
    
    if not phrase_dirs:
        print("❌ No phrase folders found!")
        print(f"   Expected location: {INPUT_DIR}")
        return
    
    print(f"✓ Found {len(phrase_dirs)} phrases")
    print()
    
    all_features = []
    all_labels = []
    skipped = 0
    total_processed = 0
    
    for phrase_idx, phrase_dir in enumerate(phrase_dirs):
        phrase_name = phrase_dir.name
        image_files = list(phrase_dir.glob('*.png')) + list(phrase_dir.glob('*.jpg'))
        
        print(f"[{phrase_idx+1}/{len(phrase_dirs)}] {phrase_name}: {len(image_files)} images", end=" ")
        
        phrase_success = 0
        for img_path in image_files:
            landmarks = extract_landmarks_from_image(img_path)
            
            if landmarks is not None:
                all_features.append(landmarks)
                all_labels.append(phrase_name)
                phrase_success += 1
            else:
                skipped += 1
            
            total_processed += 1
        
        print(f"→ {phrase_success} extracted")
    
    if not all_features:
        print("\n❌ No features extracted! Check images contain visible hands.")
        return
    
    # Create DataFrame
    print()
    print("💾 Creating feature dataset...")
    feature_cols = [f'landmark_{i}' for i in range(126)]  # 63 per hand * 2 hands
    df = pd.DataFrame(all_features, columns=feature_cols)
    df['label'] = all_labels
    
    # Save to CSV
    os.makedirs('output', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print()
    print("="*60)
    print("✅ Feature Extraction Complete!")
    print("="*60)
    print(f"Total images processed: {total_processed}")
    print(f"Successful extractions: {len(df)}")
    print(f"Skipped (no hand): {skipped}")
    print(f"Success rate: {len(df)/total_processed*100:.1f}%")
    print(f"Phrases: {df['label'].nunique()}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("📊 Samples per phrase:")
    counts = df['label'].value_counts()
    print(f"   Min: {counts.min()} | Max: {counts.max()} | Mean: {counts.mean():.1f}")
    print()
    print("▶ Next step: Run train_phrase_model.py")

if __name__ == "__main__":
    main()
