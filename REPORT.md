# Project Report: Handwritten Digit Recognizer

## Aim
To develop a machine-learning system that recognizes handwritten numerical digits.

## Objective
The objective is to train and evaluate a classifier capable of identifying digits from 0 to 9 from image pixel data.

## Technologies Used
Python, scikit-learn, Matplotlib and Joblib.

## Dataset
The project uses the handwritten digits dataset included with scikit-learn. Each sample is an 8×8 grayscale image representing a digit from 0 to 9.

## Methodology
The dataset is loaded and divided into training and testing data. The feature values are standardized, after which a K-Nearest Neighbors classifier is trained. The trained model is tested on unseen samples and evaluated using accuracy, a classification report and a confusion matrix.

## Algorithm
K-Nearest Neighbors predicts the class of an unseen sample by examining the classes of its nearest training samples. In this implementation, three nearest neighbors are used.

## Output
The program displays model accuracy, precision, recall, F1-score and confusion matrix. It also creates a visual image showing predicted and actual digits.

## Conclusion
The project demonstrates an end-to-end image-classification workflow including dataset loading, preprocessing, training, prediction, evaluation, visualization and model saving.
