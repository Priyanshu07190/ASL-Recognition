"""
ASL Sign Language Recognition - Real-Time Detection with SVM Model
Displays predictions with confidence scores using webcam feed
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ============= LOAD SVM MODEL AND SCALER =============
print("Loading SVM model and scaler...")
with open('output/asl_model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('output/asl_scaler_svm.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✅ SVM Model and Scaler loaded successfully!")
print(f"   Classes: {model.classes_}")

# ============= MEDIAPIPE SETUP =============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower threshold for better detection
    min_tracking_confidence=0.3    # Lower threshold for better tracking
)

# ============= PREDICTION SETTINGS =============
PREDICTION_THRESHOLD = 0.3  # Lower threshold to show more predictions
SMOOTHING_FRAMES = 3  # Reduced smoothing for faster response
prediction_history = []

def get_hand_bbox(hand_landmarks, image_shape):
    """Calculate bounding box around hand with padding"""
    h, w = image_shape[:2]
    
    # Get all landmark coordinates
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    
    # Calculate bounding box
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding (30%)
    padding_x = int((x_max - x_min) * 0.3)
    padding_y = int((y_max - y_min) * 0.3)
    
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(w, x_max + padding_x)
    y_max = min(h, y_max + padding_y)
    
    return (x_min, y_min, x_max, y_max)

def extract_landmarks(hand_landmarks):
    """Extract normalized landmark coordinates and bounding box"""
    if hand_landmarks is None:
        return None, None, None
    
    # Extract x, y, z coordinates for all 21 landmarks
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks).reshape(1, -1), hand_landmarks, None

def smooth_prediction(new_prediction):
    """Smooth predictions over multiple frames"""
    prediction_history.append(new_prediction)
    if len(prediction_history) > SMOOTHING_FRAMES:
        prediction_history.pop(0)
    
    # Return most common prediction
    if len(prediction_history) >= 3:
        from collections import Counter
        return Counter(prediction_history).most_common(1)[0][0]
    return new_prediction

def draw_info(image, prediction, confidence, top_predictions, fps, hand_bbox=None, hand_crop=None):
    """Draw prediction info and hand region on image"""
    h, w = image.shape[:2]
    
    # Draw FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw hand bounding box
    if hand_bbox:
        x_min, y_min, x_max, y_max = hand_bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, "Hand Region", (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create info panel background
    panel_height = 180
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    if prediction and confidence > PREDICTION_THRESHOLD:
        # Main prediction
        cv2.putText(image, f"Sign: {prediction}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.putText(image, f"Confidence: {confidence*100:.1f}%", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top 3 predictions
        cv2.putText(image, "Top 3 Predictions:", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        for i, (label, conf) in enumerate(top_predictions[:3]):
            y_pos = 140 + i * 25
            bar_width = int(conf * 300)
            cv2.rectangle(image, (120, y_pos - 15), (120 + bar_width, y_pos - 5),
                         (0, 255, 0) if i == 0 else (100, 100, 255), -1)
            cv2.putText(image, f"{label}: {conf*100:.0f}%", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(image, "No hand detected or low confidence", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show hand crop preview (bottom right)
    if hand_crop is not None:
        preview_size = 150
        hand_resized = cv2.resize(hand_crop, (preview_size, preview_size))
        
        y_offset = h - preview_size - 10
        x_offset = w - preview_size - 10
        
        cv2.rectangle(image, (x_offset - 2, y_offset - 2),
                     (x_offset + preview_size + 2, y_offset + preview_size + 2),
                     (255, 255, 255), 2)
        
        image[y_offset:y_offset + preview_size, x_offset:x_offset + preview_size] = hand_resized
        
        cv2.putText(image, "Hand Preview", (x_offset, y_offset - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

# ============= MAIN LOOP =============
def main():
    print("\n" + "="*60)
    print("ASL Real-Time Recognition - SVM Model")
    print("="*60)
    print("Controls:")
    print("  Q or ESC - Quit")
    print("  SPACE    - Pause/Resume")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    paused = False
    prev_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = hands.process(image_rgb)
                
                prediction = None
                confidence = 0
                top_predictions = []
                hand_bbox = None
                hand_crop = None
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmarks
                    landmarks, _, _ = extract_landmarks(hand_landmarks)
                    
                    if landmarks is not None:
                        # SCALE FEATURES (CRITICAL FOR SVM!)
                        landmarks_scaled = scaler.transform(landmarks)
                        
                        # Make prediction
                        prediction_class = model.predict(landmarks_scaled)[0]
                        
                        # Get probabilities
                        probabilities = model.predict_proba(landmarks_scaled)[0]
                        confidence = probabilities.max()
                        
                        # Get top 3 predictions
                        top_indices = np.argsort(probabilities)[-3:][::-1]
                        top_predictions = [(model.classes_[i], probabilities[i]) 
                                         for i in top_indices]
                        
                        # Smooth prediction
                        prediction = smooth_prediction(prediction_class)
                        
                        # Get hand bounding box
                        hand_bbox = get_hand_bbox(hand_landmarks, frame.shape)
                        
                        # Extract hand crop
                        x_min, y_min, x_max, y_max = hand_bbox
                        hand_crop = frame[y_min:y_max, x_min:x_max]
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # Draw info overlay
                frame = draw_info(frame, prediction, confidence, top_predictions, 
                                fps, hand_bbox, hand_crop)
            
            # Display frame
            cv2.imshow('ASL Recognition - SVM Model (Press Q to quit)', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n✅ Application closed")

if __name__ == "__main__":
    main()
