"""
ASL Word Builder with SVM Model
Captures letters sequentially and builds words with autocorrect
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import Counter

# ============= LOAD SVM MODEL AND SCALER =============
print("Loading SVM model and scaler...")
with open('output/asl_model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('output/asl_scaler_svm.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✅ SVM Model loaded successfully!")

# ============= MEDIAPIPE SETUP =============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# ============= WORD BUILDING SETTINGS =============
CONFIDENCE_THRESHOLD = 0.6  # Higher threshold for word building
HOLD_FRAMES = 15  # Frames to hold sign before capturing
COOLDOWN_FRAMES = 20  # Frames to wait before next letter
prediction_buffer = []
current_word = []
hold_counter = 0
cooldown_counter = 0
last_letter = None

# Common ASL words for autocorrect
COMMON_WORDS = {
    'CAT', 'DOG', 'HELLO', 'HELP', 'YES', 'NO', 'PLEASE', 'THANK', 'YOU',
    'SORRY', 'WATER', 'FOOD', 'LOVE', 'FAMILY', 'FRIEND', 'HOME', 'WORK',
    'SCHOOL', 'GOOD', 'BAD', 'HOT', 'COLD', 'BIG', 'SMALL', 'MORE', 'STOP'
}   

def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def autocorrect_word(word):
    """Find closest matching word from common words"""
    if not word:
        return None
    
    word_str = ''.join(word)
    
    # Exact match
    if word_str in COMMON_WORDS:
        return word_str
    
    # Find closest match
    closest = None
    min_distance = float('inf')
    
    for common_word in COMMON_WORDS:
        distance = levenshtein_distance(word_str, common_word)
        if distance < min_distance and distance <= 2:  # Max 2 edits
            min_distance = distance
            closest = common_word
    
    return closest

def extract_landmarks(hand_landmarks):
    """Extract normalized landmark coordinates"""
    if hand_landmarks is None:
        return None
    
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks).reshape(1, -1)

def draw_word_builder_ui(frame, prediction, confidence, fps):
    """Draw word building interface"""
    h, w = frame.shape[:2]
    
    # Create dark overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Current word being built
    word_str = ' '.join(current_word) if current_word else '(empty)'
    cv2.putText(frame, f"Word: {word_str}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Autocorrect suggestion
    suggestion = autocorrect_word(current_word)
    if suggestion and suggestion != ''.join(current_word):
        cv2.putText(frame, f"Suggestion: {suggestion}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Current sign detection
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        progress = min(hold_counter / HOLD_FRAMES, 1.0)
        color = (0, 255, 0) if progress == 1.0 else (0, 165, 255)
        
        cv2.putText(frame, f"Detecting: {prediction}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = int(progress * 300)
        cv2.rectangle(frame, (10, 150), (10 + bar_width, 170), color, -1)
        cv2.rectangle(frame, (10, 150), (310, 170), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Show a sign...", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    # Cooldown indicator
    if cooldown_counter > 0:
        cooldown_text = f"Cooldown: {cooldown_counter}/{COOLDOWN_FRAMES}"
        cv2.putText(frame, cooldown_text, (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Instructions at bottom
    overlay_bottom = frame.copy()
    cv2.rectangle(overlay_bottom, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bottom, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "SPACE: Add letter  |  BACKSPACE: Delete  |  ENTER: New word  |  Q: Quit",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Hold sign steady for 1 second to capture letter",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    global hold_counter, cooldown_counter, last_letter, current_word
    
    print("\n" + "="*60)
    print("ASL Word Builder - SVM Model")
    print("="*60)
    print("Instructions:")
    print("  1. Show a letter sign and hold steady")
    print("  2. When progress bar fills, letter is captured")
    print("  3. Build words letter by letter")
    print("  4. SPACE - Manual add | BACKSPACE - Delete | ENTER - New word")
    print("  5. Q or ESC - Quit")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            prediction = None
            confidence = 0
            
            # Handle cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
            
            if results.multi_hand_landmarks and cooldown_counter == 0:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract and predict
                landmarks = extract_landmarks(hand_landmarks)
                if landmarks is not None:
                    landmarks_scaled = scaler.transform(landmarks)
                    prediction_class = model.predict(landmarks_scaled)[0]
                    probabilities = model.predict_proba(landmarks_scaled)[0]
                    confidence = probabilities.max()
                    
                    # Filter out 'nothing' and 'space'
                    if prediction_class not in ['nothing', 'space']:
                        prediction = prediction_class
                        
                        # Check if holding same sign
                        if prediction == last_letter and confidence > CONFIDENCE_THRESHOLD:
                            hold_counter += 1
                            
                            # Capture letter after holding
                            if hold_counter >= HOLD_FRAMES:
                                current_word.append(prediction)
                                print(f"✓ Captured: {prediction} | Word: {' '.join(current_word)}")
                                
                                # Check for autocorrect
                                suggestion = autocorrect_word(current_word)
                                if suggestion:
                                    print(f"  💡 Suggestion: {suggestion}")
                                
                                hold_counter = 0
                                cooldown_counter = COOLDOWN_FRAMES
                                last_letter = None
                        else:
                            # Different sign, reset counter
                            hold_counter = 0
                            last_letter = prediction
            else:
                hold_counter = 0
                last_letter = None
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            
            # Draw UI
            frame = draw_word_builder_ui(frame, prediction, confidence, fps)
            
            cv2.imshow('ASL Word Builder - SVM (Q to quit)', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == 8 or key == 127:  # BACKSPACE or DELETE
                if current_word:
                    removed = current_word.pop()
                    print(f"✗ Deleted: {removed} | Word: {' '.join(current_word)}")
            elif key == 13:  # ENTER
                if current_word:
                    final_word = ''.join(current_word)
                    suggestion = autocorrect_word(current_word)
                    print(f"\n🎯 Final Word: {final_word}")
                    if suggestion and suggestion != final_word:
                        print(f"   Did you mean: {suggestion}?")
                    print()
                    current_word = []
            elif key == ord(' '):  # SPACE - manual add
                if prediction and confidence > CONFIDENCE_THRESHOLD:
                    current_word.append(prediction)
                    print(f"✓ Added: {prediction} | Word: {' '.join(current_word)}")
                    cooldown_counter = COOLDOWN_FRAMES // 2
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\n✅ Application closed")

if __name__ == "__main__":
    main()
