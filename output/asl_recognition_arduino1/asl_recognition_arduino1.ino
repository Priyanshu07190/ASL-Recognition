/*
 * ASL Sign Language Recognition - Arduino Sketch
 * For Arduino Nano 33 BLE Sense with OV7675 Camera
 */

#include <Arduino_OV767X.h>
#include "asl_model.h"

// Create classifier instance
Eloquent::ML::Port::RandomForest classifier;

const int LED_PIN = LED_BUILTIN;
const int BUFFER_SIZE = 63;
float features[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  pinMode(LED_PIN, OUTPUT);
  
  Serial.println("====================================");
  Serial.println("ASL Sign Language Recognition");
  Serial.println("====================================");
  Serial.println();
  
  // Initialize camera
  Serial.println("Initializing OV7675 camera...");
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("ERROR: Camera initialization failed!");
    Serial.println("Check camera connections.");
    // Continue anyway - can still receive landmarks via serial
  } else {
    Serial.println("Camera initialized successfully!");
  }
  
  Serial.println();
  Serial.println("Model loaded and ready!");
  Serial.println("Waiting for hand landmark data...");
  Serial.println("Format: LANDMARKS:x1,y1,z1,x2,y2,z2,...");
  Serial.println();
  
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input == "CAPTURE") {
      captureAndSendImage();
    }
    else if (input.startsWith("LANDMARKS:")) {
      input.remove(0, 10);
      int index = 0;
      int lastComma = -1;
      
      for (int i = 0; i <= input.length() && index < BUFFER_SIZE; i++) {
        if (i == input.length() || input.charAt(i) == ',') {
          String value = input.substring(lastComma + 1, i);
          features[index++] = value.toFloat();
          lastComma = i;
        }
      }
      
      if (index == BUFFER_SIZE) {
        // Print received coordinates
        Serial.println("\n--- Received Landmarks ---");
        for (int i = 0; i < BUFFER_SIZE; i += 3) {
          Serial.print("Point ");
          Serial.print(i/3);
          Serial.print(": x=");
          Serial.print(features[i], 6);
          Serial.print(", y=");
          Serial.print(features[i+1], 6);
          Serial.print(", z=");
          Serial.println(features[i+2], 6);
        }
        
        char prediction = predict(features);
        
        Serial.print("\nPREDICTION: ");
        Serial.println(prediction);
        Serial.println("-------------------------\n");
        
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
      } else {
        Serial.println("ERROR: Invalid number of features");
      }
    }
  }
  
  delay(50);
}

void captureAndSendImage() {
  Serial.println("Capturing image...");
  
  int width = Camera.width();
  int height = Camera.height();
  int bytesPerPixel = Camera.bytesPerPixel();
  int imageSize = width * height * bytesPerPixel;
  
  // Allocate buffer for image
  uint8_t* buffer = new uint8_t[imageSize];
  
  // Read frame into buffer
  Camera.readFrame(buffer);
  
  Serial.print("IMAGE_START:");
  Serial.print(width);
  Serial.print(",");
  Serial.print(height);
  Serial.print(",");
  Serial.println(bytesPerPixel);
  
  digitalWrite(LED_PIN, HIGH);
  
  // Send image data
  Serial.write(buffer, imageSize);
  Serial.println();
  Serial.println("IMAGE_END");
  
  digitalWrite(LED_PIN, LOW);
  
  // Free buffer
  delete[] buffer;
  
  Serial.println("Image sent successfully!");
}

char predict(float* input) {
  const char* label = classifier.predictLabel(input);
  return label[0];  // Return first character of the label (A-Z)
}
