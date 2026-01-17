"""
ASL Phrase Recognition - Real-Time Detection
Recognizes 44 complete phrases in one shot
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ============= LOAD MODEL AND SCALER =============
print("Loading phrase recognition model...")
with open('output/asl_phrase_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('output/asl_phrase_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"   Phrases: {len(model.classes_)}")
print()

# ============= MEDIAPIPE SETUP =============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect both hands for phrases
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# ============= PREDICTION SETTINGS =============
CONFIDENCE_THRESHOLD = 0.4  # Lower threshold to see more predictions
SMOOTHING_FRAMES = 5
prediction_history = []

def extract_landmarks(multi_hand_landmarks):
    """Extract normalized landmark coordinates from both hands"""
    if multi_hand_landmarks is None or len(multi_hand_landmarks) == 0:
        return None
    
    landmarks = []
    
    # Extract landmarks for up to 2 hands
    for hand_idx in range(min(2, len(multi_hand_landmarks))):
        hand_landmarks = multi_hand_landmarks[hand_idx]
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # If only 1 hand detected, pad with zeros
    if len(multi_hand_landmarks) == 1:
        landmarks.extend([0.0] * 63)
    
    return np.array(landmarks).reshape(1, -1) if len(landmarks) == 126 else None

def smooth_prediction(new_prediction):
    """Smooth predictions over multiple frames"""
    prediction_history.append(new_prediction)
    if len(prediction_history) > SMOOTHING_FRAMES:
        prediction_history.pop(0)
    
    if len(prediction_history) >= 3:
        from collections import Counter
        return Counter(prediction_history).most_common(1)[0][0]
    return new_prediction

def draw_phrase_ui(frame, prediction, confidence, top_predictions, fps):
    """Draw phrase recognition interface"""
    h, w = frame.shape[:2]
    
    # Top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        # Main prediction
        cv2.putText(frame, f"Phrase: {prediction.upper()}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Top 3 predictions
        cv2.putText(frame, "Top 3 Phrases:", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        for i, (label, conf) in enumerate(top_predictions[:3]):
            y_pos = 165 + i * 30
            
            # Confidence bar
            bar_width = int(conf * 400)
            color = (0, 255, 0) if i == 0 else (100, 150, 255)
            cv2.rectangle(frame, (180, y_pos - 18), (180 + bar_width, y_pos - 8),
                         color, -1)
            
            # Text
            text = f"{i+1}. {label}: {conf*100:.0f}%"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Show a phrase sign...", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cv2.putText(frame, "Hold your hand steady", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Instructions at bottom
    overlay_bottom = frame.copy()
    cv2.rectangle(overlay_bottom, (0, h - 100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bottom, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "Recognized Phrases:", (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Show some example phrases
    examples = ["again", "agree", "please", "thank you", "how are you", "i need help", 
                "happy birthday", "good morning", "home", "stop"]
    example_text = ", ".join(examples[:8])
    cv2.putText(frame, example_text, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.putText(frame, "Press Q to quit  |  SPACE to pause", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    print("\n" + "="*60)
    print("ASL Phrase Recognition - Real-Time")
    print("="*60)
    print("Recognizes 44 common phrases:")
    print("  - Greetings: hello, good morning, happy birthday")
    print("  - Actions: please, thank you, help, stop, wait")
    print("  - Questions: how are you, where, what")
    print("  - And 40+ more!")
    print()
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
                    break
                
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                prediction = None
                confidence = 0
                top_predictions = []
                
                if results.multi_hand_landmarks:
                    # Draw landmarks for all detected hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract and predict from both hands
                    landmarks = extract_landmarks(results.multi_hand_landmarks)
                    if landmarks is not None:
                        landmarks_scaled = scaler.transform(landmarks)
                        prediction_class = model.predict(landmarks_scaled)[0]
                        probabilities = model.predict_proba(landmarks_scaled)[0]
                        confidence = probabilities.max()
                        
                        # Get top 3
                        top_indices = np.argsort(probabilities)[-3:][::-1]
                        top_predictions = [(model.classes_[i], probabilities[i]) 
                                         for i in top_indices]
                        
                        # Smooth prediction
                        prediction = smooth_prediction(prediction_class)
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
                prev_time = curr_time
                
                # Draw UI
                frame = draw_phrase_ui(frame, prediction, confidence, top_predictions, fps)
            
            cv2.imshow('ASL Phrase Recognition (Q to quit)', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
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
