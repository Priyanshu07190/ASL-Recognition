# 🤟 ASL Recognition System

Real-time American Sign Language (ASL) alphabet and phrase recognition using Computer Vision and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

---

## Overview

This project implements a complete ASL recognition pipeline that detects and classifies hand gestures in real-time. It supports both **individual letters (A-Z)** and **common phrases** using MediaPipe for hand landmark detection and SVM/Random Forest classifiers for recognition.

### Why This Project?
- **Accessibility**: Helps bridge communication gaps for the deaf and hard-of-hearing community
- **Real-time Performance**: Processes video at 30+ FPS for smooth interaction
- **Multi-Model Support**: Includes both traditional ML (SVM) and ensemble methods
- **Arduino Deployment**: Converts models to C++ for embedded systems

---

## Features

✅ **Real-time Hand Tracking** - Uses MediaPipe for robust 21-landmark hand detection  
✅ **Alphabet Recognition** - Recognizes all 26 letters + space + delete  
✅ **Phrase Recognition** - Detects 45+ common ASL phrases (2-handed gestures)  
✅ **Multiple Models** - SVM and Random Forest classifiers with 95%+ accuracy  
✅ **Word Builder** - Interactive app to spell words letter-by-letter  
✅ **Arduino Integration** - Export models for microcontroller deployment  
✅ **Confusion Matrix Visualization** - Detailed performance analysis

---

## Tech Stack

### Machine Learning & CV
- **Python 3.8+** - Core programming language
- **OpenCV** - Image processing and webcam integration
- **MediaPipe** - Hand landmark detection
- **Scikit-learn** - SVM, Random Forest classifiers
- **NumPy & Pandas** - Data manipulation

### Deployment
- **Arduino C++** - Embedded model deployment
- **Matplotlib** - Visualization and analysis

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Priyanshu07190/ASL-Recognition.git
cd ASL-Recognition
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- opencv-python>=4.8.0
- mediapipe>=0.10.0
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- Pillow>=10.0.0
- pyserial>=3.5

---

## Usage

### 1. Train Your Own Model (Optional)

If you have your own dataset:

```bash
# Extract features from images
python 1_extract_features.py

# Train SVM model
python 2_train_model_svm.py

# OR train Random Forest model
python 2_train_model.py
```

### 2. Run Real-Time Alphabet Recognition

```bash
python app_realtime_svm.py
```
- Shows live webcam feed with hand landmarks
- Displays recognized letter in real-time
- Press **'q'** to quit

### 3. Run Word Builder

```bash
python app_word_builder_svm.py
```
- Spell words by showing letters one by one
- Built-in word display and deletion support
- Press **Space** to add letter, **Del** to remove

### 4. Run Phrase Recognition

```bash
python app_phrase_recognition.py
```
- Supports 45+ ASL phrases
- Works with 2-handed gestures
- Real-time phrase detection

### 5. Arduino Deployment

```bash
# Convert model to Arduino C++ header file
python 3_convert_arduino.py

# Upload arduino_camera_bridge/arduino_camera_bridge.ino to your Arduino
```

---

## Project Structure

```
ASL-Recognition/
├── 1_extract_features.py           # Extract hand landmarks from images
├── 2_train_model.py                # Train Random Forest model
├── 2_train_model_svm.py            # Train SVM model
├── 3_convert_arduino.py            # Convert model to Arduino
├── app_realtime_svm.py             # Real-time letter recognition
├── app_word_builder_svm.py         # Interactive word spelling
├── app_phrase_recognition.py       # Phrase detection
├── app_webcam_arduino.py           # Arduino webcam integration
├── extract_phrase_features.py      # Extract phrase features
├── train_phrase_model.py           # Train phrase model
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── arduino_camera_bridge/          # Arduino code
│   ├── arduino_camera_bridge.ino   # Main Arduino sketch
│   └── asl_model.h                 # Model weights (C array)
│
└── output/                         # Generated files
    ├── asl_model.pkl               # Trained alphabet model
    ├── asl_model_svm.pkl           # SVM alphabet model
    ├── asl_phrase_model.pkl        # Phrase model
    ├── asl_scaler_svm.pkl          # Feature scaler
    ├── classification_report.txt   # Performance metrics
    ├── confusion_matrix.png        # Visualization
    └── ...
```

---

## Model Performance

### Alphabet Recognition (SVM)
- **Accuracy**: 96.5%
- **Classes**: 28 (A-Z + space + delete)
- **Features**: 63 (21 landmarks × 3 coordinates)
- **Training Samples**: 87,000+ images

### Phrase Recognition
- **Accuracy**: 94.2%
- **Phrases**: 45
- **Features**: 126 (42 landmarks for 2 hands)
- **Real-time FPS**: 30+

*See `output/classification_report.txt` for detailed metrics*

---

## How It Works

### 1. Hand Landmark Detection
- MediaPipe detects 21 key points on each hand
- Extracts (x, y, z) coordinates for each landmark
- Normalizes coordinates relative to wrist position

### 2. Feature Engineering
- Flatten 21 landmarks into 63-dimensional feature vector
- Scale features using StandardScaler
- Handle both 1-hand (letters) and 2-hand (phrases) gestures

### 3. Classification
- **SVM (RBF kernel)**: Fast, accurate for letter recognition
- **Random Forest**: Robust for phrase recognition
- Real-time inference at 30+ FPS

### 4. Post-Processing
- Smoothing filter to reduce jitter
- Confidence threshold for reliable predictions
- Visual feedback with bounding boxes and labels

---

## Future Improvements

- [ ] Add more phrases (100+ common ASL phrases)
- [ ] Implement sentence construction with grammar
- [ ] Add data augmentation for better generalization
- [ ] Deploy as mobile app (Flutter/React Native)
- [ ] Add sign language translation (ASL ↔ Text)
- [ ] Support for continuous sign recognition (not just static)
- [ ] Multi-language support (BSL, ISL, etc.)
- [ ] Cloud deployment with REST API

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## �‍💻 Author

**Priyanshu**  
GitHub: [@Priyanshu07190](https://github.com/Priyanshu07190)

---

## 🙏 Acknowledgments

- MediaPipe team for the hand tracking solution
- Kaggle ASL dataset contributors
- Scikit-learn community

---

**⭐ If you find this project helpful, please give it a star!**

