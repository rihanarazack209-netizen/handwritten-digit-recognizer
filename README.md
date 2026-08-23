# Handwritten Digit Recognizer

## Project Overview
This project recognizes handwritten digits using machine learning. It uses the handwritten digit dataset built into scikit-learn, so no separate dataset download is required.

## Python
Designed to run on Python 3.14 with current package versions.

## Algorithm
K-Nearest Neighbors (KNN) with feature standardization.

## Features
- Loads handwritten digit images
- Splits data into training and testing sets
- Trains a KNN classifier
- Prints model accuracy
- Prints a classification report
- Prints a confusion matrix
- Generates a visual prediction image
- Saves the trained model

## Run
Open a terminal in this folder and run:

```powershell
python -m pip install -r requirements.txt
python digit_recognizer.py
```

## Files Generated After Running
- `digit_predictions.png`
- `digit_recognizer_model.pkl`

## Screenshots for Submission
Take screenshots of:
1. Test Accuracy
2. Classification Report
3. Prediction image containing handwritten digits
