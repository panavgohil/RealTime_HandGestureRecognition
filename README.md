# Real-Time Hand Gesture Recognition

## Overview

A real-time hand gesture recognition system built using MediaPipe, OpenCV, and Machine Learning. The project detects hand landmarks from a webcam feed, extracts relative coordinate features, and classifies gestures using a Random Forest classifier.

## Features

* Real-time webcam-based gesture recognition
* MediaPipe hand landmark detection
* Relative-coordinate feature extraction
* Random Forest classifier
* Confidence score display
* Custom dataset collection pipeline

## Supported Gestures

* Open Palm
* Fist
* Point
* Thumbs Up

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Pandas
* Scikit-learn
* Joblib

## Project Workflow

Webcam Feed → MediaPipe Hand Detection → Landmark Extraction → Feature Engineering → Random Forest Classification → Real-Time Prediction

## Dataset

A custom dataset was created using MediaPipe hand landmarks.

* 21 hand landmarks detected per frame
* 42 features extracted (relative x and y coordinates)
* Balanced dataset across all gesture classes

## Model Performance

Random Forest Classifier

* Accuracy: 98.75%

Classification Report:

| Gesture   | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Fist      | 1.00      | 0.98   | 0.99     |
| Open      | 1.00      | 0.97   | 0.99     |
| Point     | 0.97      | 1.00   | 0.99     |
| Thumbs Up | 0.98      | 1.00   | 0.99     |

## Installation

```bash
git clone https://github.com/panavgohil/RealTime_HandGestureRecognition.git
cd RealTime_HandGestureRecognition

pip install -r requirements.txt
```

## Usage

### Collect Dataset

```bash
python collect_data.py
```

### Train Model

```bash
python train_model.py
```

### Run Real-Time Prediction

```bash
python predict.py
```

## Future Improvements

* Additional hand gestures
* Gesture-controlled applications
* Gesture-controlled robot navigation
* Deep learning based classification
* Dynamic gesture recognition

## Author

Panav Gohil

DTU '29 | Electronics Engineering | ML, Computer Vision & Embedded Systems
