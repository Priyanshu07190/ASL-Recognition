"""Test script to send landmarks to Arduino via Serial"""

import serial
import time
import pandas as pd

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
TEST_CSV = "output/asl_features.csv"

def send_landmarks_to_arduino(ser, landmarks):
    landmark_str = "LANDMARKS:" + ",".join([str(x) for x in landmarks])
    ser.write((landmark_str + "\n").encode())
    time.sleep(0.1)
    
    if ser.in_waiting:
        response = ser.readline().decode().strip()
        return response
    return None

def test_with_dataset():
    print("Loading test dataset...")
    df = pd.read_csv(TEST_CSV)
    
    print(f"Attempting to connect to Arduino on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Connected to Arduino\n")
        
        samples = df.sample(n=min(10, len(df)))
        
        print("Testing predictions:\n")
        correct = 0
        total = 0
        
        for idx, row in samples.iterrows():
            features = row[:-1].values
            true_label = row.iloc[-1]
            
            response = send_landmarks_to_arduino(ser, features)
            
            if response and "Prediction:" in response:
                predicted = response.split("Prediction:")[1].strip()
                is_correct = predicted == true_label
                correct += is_correct
                total += 1
                
                status = "OK" if is_correct else "FAIL"
                print(f"{status} True: {true_label}, Predicted: {predicted}")
            else:
                print(f"WARNING: No response from Arduino")
            
            time.sleep(0.5)
        
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        
        ser.close()
        print("\nTest complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check that Arduino is connected")
        print("2. Verify the correct port")
        print("3. Close Arduino IDE Serial Monitor if open")
        print("4. Install pyserial: pip install pyserial")

if __name__ == "__main__":
    test_with_dataset()
