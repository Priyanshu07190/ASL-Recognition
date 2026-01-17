"""
ASL Recognition - PC Webcam + Arduino Prediction
Uses PC webcam for better image quality, Arduino for prediction
"""

import serial
import numpy as np
import cv2
import time
import mediapipe as mp

SERIAL_PORT = 'COM11'
BAUD_RATE = 115200

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def connect_arduino():
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)
    
    print("\n--- Arduino Initialization ---")
    for _ in range(15):
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  {line}")
    
    return ser

def send_landmarks_to_arduino(ser, landmarks):
    landmarks_str = ','.join([f'{x:.6f}' for x in landmarks])
    command = f"LANDMARKS:{landmarks_str}\n"
    
    ser.write(command.encode())
    ser.flush()
    
    # Wait for prediction
    time.sleep(0.1)
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("PREDICTION:"):
            prediction = line.split(':')[1].strip()
            return prediction
        elif "ERROR" in line:
            print(f"Arduino Error: {line}")
            return None
    
    return None

def main():
    print("="*60)
    print("ASL Recognition - PC Webcam + Arduino Prediction")
    print("="*60)
    
    try:
        # Connect to Arduino
        ser = connect_arduino()
        
        # Open webcam
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        print("Webcam opened successfully!")
        print("\n" + "="*60)
        print("Press 'SPACE' to capture and recognize")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        count = 0
        last_prediction = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks
            display_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                
                # Show "Hand Detected" indicator
                cv2.putText(display_frame, "Hand Detected - Press SPACE", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No Hand - Show hand to camera", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            
            # Show last prediction
            if last_prediction:
                cv2.putText(display_frame, f"Last Prediction: {last_prediction}", 
                          (10, display_frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1.2, (0, 255, 0), 3)
            
            cv2.imshow('ASL Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar
                if results.multi_hand_landmarks:
                    count += 1
                    print(f"\n{'='*60}")
                    print(f"Recognition #{count}")
                    print(f"{'='*60}")
                    
                    # Extract landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    print(f"[1/2] Extracted {len(landmarks)} features")
                    
                    # Send to Arduino
                    print(f"[2/2] Sending to Arduino for prediction...")
                    prediction = send_landmarks_to_arduino(ser, landmarks)
                    
                    if prediction:
                        print(f"\n✓ Predicted Sign: {prediction}")
                        last_prediction = prediction
                        
                        # Show prediction on frame
                        pred_frame = display_frame.copy()
                        cv2.putText(pred_frame, f"Prediction: {prediction}", 
                                  (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  2.5, (0, 255, 0), 5)
                        cv2.imshow('ASL Recognition', pred_frame)
                        cv2.waitKey(2000)  # Show for 2 seconds
                    else:
                        print(f"\n✗ No prediction received from Arduino")
                    
                    print(f"{'='*60}\n")
                else:
                    print("\n✗ No hand detected! Show your hand to the camera.\n")
            
            elif key == ord('q'):
                print("\n\nExiting...")
                break
        
        cap.release()
        ser.close()
        cv2.destroyAllWindows()
        hands.close()
        
        print(f"\nTotal recognitions: {count}")
        print("Done!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
