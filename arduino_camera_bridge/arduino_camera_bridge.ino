/*
 * ASL Sign Language Recognition - Arduino Sketch with Camera Module
 * For Arduino Nano 33 BLE Sense + OV7675/OV7670 Camera Module
 * 
 * Camera captures image → sends to PC → PC extracts landmarks → 
 * Arduino predicts letter using Random Forest
 */

#include "asl_model.h"
#include <Arduino_OV767X.h>

// Use the namespace to avoid typing it repeatedly
using namespace Eloquent::ML::Port;

const int LED_PIN = LED_BUILTIN;
const int BUFFER_SIZE = 63;
float features[BUFFER_SIZE];

// Camera settings
const int IMAGE_WIDTH = 176;   // QCIF resolution
const int IMAGE_HEIGHT = 144;
const int BYTES_PER_PIXEL = 2; // RGB565 format

// Create classifier instance
RandomForest classifier;

// Communication protocol
enum CommandType {
  CMD_CAPTURE = 'C',      // PC requests image capture
  CMD_IMAGE_START = 'I',  // Arduino starts sending image
  CMD_IMAGE_END = 'E',    // Arduino finished sending image
  CMD_LANDMARKS = 'L',    // PC sends landmarks
  CMD_READY = 'R',        // Arduino ready
  CMD_PREDICTION = 'P'    // Arduino sends prediction
};

bool cameraInitialized = false;

// Forward declarations
char predictSign(float* input);
void captureAndSendImage();
void handleLandmarks();

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  pinMode(LED_PIN, OUTPUT);
  
  Serial.println("====================================");
  Serial.println("ASL Recognition - Arduino Camera Mode");
  Serial.println("====================================");
  Serial.println();
  
  // Initialize classifier
  Serial.println("Initializing Random Forest classifier...");
  Serial.println("Classifier ready!");
  
  // Initialize camera
  Serial.println();
  Serial.println("Initializing camera module...");
  
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("ERROR: Failed to initialize camera!");
    Serial.println("Check camera connections:");
    Serial.println("  - VCC -> 3.3V");
    Serial.println("  - GND -> GND");
    Serial.println("  - SDA -> A4");
    Serial.println("  - SCL -> A5");
    Serial.println();
    Serial.println("Will continue anyway (PC can use webcam)...");
    cameraInitialized = false;
  } else {
    Serial.println("Camera initialized successfully!");
    cameraInitialized = true;
  }
  
  Serial.println();
  Serial.println("System ready!");
  Serial.println("Waiting for PC connection...");
  Serial.println();
  
  // Blink LED to indicate ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  // Send ready signal
  Serial.write(CMD_READY);
  Serial.flush();
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case CMD_CAPTURE:
        // PC requests image capture
        if (cameraInitialized) {
          captureAndSendImage();
        } else {
          // Camera not available, PC will use webcam
          Serial.write(CMD_IMAGE_END); // Signal to use PC webcam
        }
        break;
        
      case CMD_LANDMARKS:
        // PC sends landmarks for prediction
        handleLandmarks();
        break;
        
      default:
        // Ignore unknown commands
        break;
    }
  }
  
  delay(10);
}

void captureAndSendImage() {
  digitalWrite(LED_PIN, HIGH);
  
  // Read frame from camera
  Camera.readFrame(NULL); // Skip first frame (often corrupted)
  delay(10);
  
  // Read actual frame
  uint8_t* buffer = (uint8_t*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL);
  if (buffer == NULL) {
    Serial.println("ERROR: Memory allocation failed!");
    digitalWrite(LED_PIN, LOW);
    return;
  }
  
  Camera.readFrame(buffer);
  
  // Send image start marker
  Serial.write(CMD_IMAGE_START);
  
  // Send image dimensions
  Serial.write((uint8_t)(IMAGE_WIDTH & 0xFF));
  Serial.write((uint8_t)(IMAGE_WIDTH >> 8));
  Serial.write((uint8_t)(IMAGE_HEIGHT & 0xFF));
  Serial.write((uint8_t)(IMAGE_HEIGHT >> 8));
  
  // Send image data in chunks
  int totalBytes = IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL;
  int chunkSize = 256;
  
  for (int i = 0; i < totalBytes; i += chunkSize) {
    int bytesToSend = min(chunkSize, totalBytes - i);
    Serial.write(&buffer[i], bytesToSend);
    Serial.flush();
    delay(1); // Small delay to prevent buffer overflow
  }
  
  // Send image end marker
  Serial.write(CMD_IMAGE_END);
  Serial.flush();
  
  free(buffer);
  digitalWrite(LED_PIN, LOW);
}

void handleLandmarks() {
  // Read the newline-terminated landmark data
  String input = Serial.readStringUntil('\n');
  
  int index = 0;
  int lastComma = -1;
  
  // Parse comma-separated values
  for (int i = 0; i <= input.length() && index < BUFFER_SIZE; i++) {
    if (i == input.length() || input.charAt(i) == ',') {
      String value = input.substring(lastComma + 1, i);
      features[index++] = value.toFloat();
      lastComma = i;
    }
  }
  
  if (index == BUFFER_SIZE) {
    // Make prediction
    char prediction = predictSign(features);
    
    // Send prediction back
    Serial.write(CMD_PREDICTION);
    Serial.print("Prediction: ");
    Serial.println(prediction);
    
    // Blink LED
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
  } else {
    Serial.println("ERROR: Invalid number of features");
  }
}

char predictSign(float* input) {
  // Use the classifier's predict method
  int classIndex = classifier.predict(input);
  return 'A' + classIndex;
}
