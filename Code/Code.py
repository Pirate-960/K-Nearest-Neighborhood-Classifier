import pandas as pd
import numpy as np
import json
from collections import Counter

# Helper Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def normalize_features(features):
    """Normalize features to a 0-1 range."""
    return (features - features.min()) / (features.max() - features.min())

# Data Preparation
def load_dataset_from_json(file_path):
    """Load dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.drop(columns=['Day'])  # Drop the 'Day' column as it's not needed
    return df

def encode_categorical_features(df):
    """Convert categorical features into numeric using one-hot encoding."""
    features = df.drop(columns=['PlayTennis'])
    target = df['PlayTennis']
    features_encoded = pd.get_dummies(features, drop_first=True)
    target_encoded = target.map({'Yes': 1, 'No': 0})
    return features_encoded, target_encoded

# k-NN Implementation
def knn_predict(test_instance, training_data, training_labels, k, distance_metric):
    distances = []
    for i in range(len(training_data)):
        distance = distance_metric(test_instance, training_data[i])
        distances.append((distance, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common

def evaluate_knn(features, labels, k, distance_metric):
    predictions = []
    for i in range(len(features)):
        test_instance = features[i]
        test_label = labels[i]
        training_data = np.delete(features, i, axis=0)
        training_labels = np.delete(labels, i, axis=0)
        predicted_label = knn_predict(test_instance, training_data, training_labels, k, distance_metric)
        predictions.append(predicted_label)
    return predictions

# Performance Metrics
def calculate_metrics(actuals, predictions):
    tp = sum((a == 1 and p == 1) for a, p in zip(actuals, predictions))
    tn = sum((a == 0 and p == 0) for a, p in zip(actuals, predictions))
    fp = sum((a == 0 and p == 1) for a, p in zip(actuals, predictions))
    fn = sum((a == 1 and p == 0) for a, p in zip(actuals, predictions))

    accuracy = (tp + tn) / len(actuals)
    confusion_matrix = [[tp, fn], [fp, tn]]
    return accuracy, confusion_matrix

# Execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    json_path = "Dataset/play_tennis.json"
    dataset = load_dataset_from_json(json_path)
    features, labels = encode_categorical_features(dataset)
    features = normalize_features(features).values
    labels = labels.values

    # Display the dataset
    print("===== Dataset =====\n")
    print(dataset.to_string(index=False))
    print("\n===== Dataset Summary =====")
    print(dataset.describe(include='all'))

    # Get user inputs
    k = int(input("\nEnter the value of k: "))
    distance_choice = input("Select distance metric (1: Euclidean, 2: Manhattan): ")
    distance_metric = euclidean_distance if distance_choice == "1" else manhattan_distance

    # Perform Leave-One-Out Cross-Validation
    predictions = evaluate_knn(features, labels, k, distance_metric)

    # Calculate performance
    accuracy, confusion_matrix = calculate_metrics(labels, predictions)
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(f"TP: {confusion_matrix[0][0]}, FN: {confusion_matrix[0][1]}")
    print(f"FP: {confusion_matrix[1][0]}, TN: {confusion_matrix[1][1]}")
