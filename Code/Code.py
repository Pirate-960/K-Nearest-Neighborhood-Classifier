import pandas as pd
import numpy as np
from collections import Counter

# Helper Functions
def euclidean_distance(x1, x2):
    """Calculates the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """Calculates the Manhattan distance between two vectors."""
    return np.sum(np.abs(x1 - x2))

# Data Preparation
def load_play_tennis_dataset():
    """Loads the Play Tennis dataset."""
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                    'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                     'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
                 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
                       'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    return pd.DataFrame(data)

def encode_categorical_features(df):
    """Encodes categorical features into numeric using one-hot encoding."""
    features = df.drop(columns=['PlayTennis'])
    target = df['PlayTennis']
    # Apply one-hot encoding to features
    features_encoded = pd.get_dummies(features, drop_first=True)
    # Map the target variable to 0 (No) and 1 (Yes)
    target_encoded = target.map({'Yes': 1, 'No': 0})
    # Convert the features to numeric type
    features_encoded = features_encoded.astype(int)
    return features_encoded, target_encoded

# Core k-NN Implementation
def knn_predict(test_instance, training_data, training_labels, k, distance_metric):
    """
    Predicts the label for a test instance using the k-NN algorithm.
    Args:
    - test_instance: The feature vector of the test instance.
    - training_data: The training feature matrix.
    - training_labels: The training labels.
    - k: Number of neighbors to consider.
    - distance_metric: Function to calculate distance (e.g., Euclidean or Manhattan).
    Returns:
    - Predicted label (0 or 1).
    """
    distances = []
    for i in range(len(training_data)):
        distance = distance_metric(test_instance, training_data[i])
        distances.append((distance, training_labels[i]))
    # Sort by distance and select the k-nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    # Return the most common label among the neighbors
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common

def evaluate_knn(features, labels, k, distance_metric):
    """
    Evaluates the k-NN classifier without Leave-One-Out Cross-Validation.
    Args:
    - features: The feature matrix.
    - labels: The target labels.
    - k: Number of neighbors to consider.
    - distance_metric: Function to calculate distance (e.g., Euclidean or Manhattan).
    Returns:
    - Accuracy of the model.
    """
    predictions = []
    for i in range(len(features)):
        test_instance = features[i]
        predicted_label = knn_predict(test_instance, features, labels, k, distance_metric)
        predictions.append(predicted_label)
    accuracy = np.mean(predictions == labels)
    return accuracy

def loocv_knn(features, labels, k, distance_metric):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) for the k-NN classifier.
    Args:
    - features: The feature matrix.
    - labels: The target labels.
    - k: Number of neighbors to consider.
    - distance_metric: Function to calculate distance (e.g., Euclidean or Manhattan).
    Returns:
    - Accuracy of the model using LOOCV.
    """
    correct_predictions = 0
    for i in range(len(features)):
        test_instance = features[i]
        test_label = labels[i]
        # Train on all data except the i-th instance
        training_data = np.delete(features, i, axis=0)
        training_labels = np.delete(labels, i, axis=0)
        predicted_label = knn_predict(test_instance, training_data, training_labels, k, distance_metric)
        if predicted_label == test_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(features)
    return accuracy

# Execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    dataset = load_play_tennis_dataset()
    features, labels = encode_categorical_features(dataset)
    features = features.values  # Convert to NumPy array for faster computation
    labels = labels.values      # Convert to NumPy array for faster computation
    
    # Parameters
    k = int(input("Enter the value of k: "))
    distance_metric = euclidean_distance  # Choose euclidean_distance or manhattan_distance

    # Solution 1: Without Leave-One-Out Cross-Validation
    accuracy_no_loocv = evaluate_knn(features, labels, k, distance_metric)
    print(f"Accuracy without LOOCV: {accuracy_no_loocv:.2f}")

    # Solution 2: With Leave-One-Out Cross-Validation
    accuracy_loocv = loocv_knn(features, labels, k, distance_metric)
    print(f"Accuracy with LOOCV: {accuracy_loocv:.2f}")