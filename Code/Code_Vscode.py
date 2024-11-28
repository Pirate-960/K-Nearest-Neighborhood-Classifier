import pandas as pd
import numpy as np
from collections import Counter
import json

# Helper Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def normalize_features(features):
    """Normalize features to a 0-1 range after ensuring all columns are numeric."""
    # Convert boolean columns to integers
    features = features.applymap(lambda x: int(x) if isinstance(x, (bool, np.bool_)) else x)
    # Perform normalization
    return (features - features.min()) / (features.max() - features.min())

# Data Preparation
def load_dataset_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def encode_categorical_features(df):
    features = df.drop(columns=['PlayTennis'])
    target = df['PlayTennis']
    features_encoded = pd.get_dummies(features, drop_first=True)
    target_encoded = target.map({'Yes': 1, 'No': 0})
    return features_encoded, target_encoded

# Core k-NN Implementation
def knn_predict(test_instance, training_data, training_labels, k, distance_metric):
    distances = []
    for i in range(len(training_data)):
        distance = distance_metric(test_instance, training_data[i])
        distances.append((distance, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common, distances[:k]

def evaluate_knn(features, labels, k, distance_metric, log_file):
    predictions = []
    actuals = []
    with open(log_file, 'w') as log:
        log.write("===== k-NN Classification Without LOOCV =====\n\n")
        log.write(f"k: {k}, Distance Metric: {distance_metric.__name__}\n")
        log.write(f"Total Instances: {len(features)}\n\n")
        print(f"===== k-NN Classification Without LOOCV =====\n")

        for i in range(len(features)):
            test_instance = features[i]
            actual_label = labels[i]
            predicted_label, neighbors = knn_predict(test_instance, features, labels, k, distance_metric)
            predictions.append(predicted_label)
            actuals.append(actual_label)

            log.write(f"--- Test Instance {i+1} ---\n")
            log.write(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}\n")
            log.write(f"Neighbors (distance, label):\n")
            print(f"Test Instance {i+1}:")
            print(f"Actual: {actual_label}, Predicted: {predicted_label}")
            print(f"Neighbors (distance, label):")
            
            for dist, lbl in neighbors:
                log.write(f"  Distance: {dist:.3f}, Label: {lbl}\n")
                print(f"  Distance: {dist:.3f}, Label: {lbl}")
            log.write("\n")
            print()

    return predictions, actuals

def loocv_knn(features, labels, k, distance_metric, log_file):
    correct_predictions = 0
    predictions = []
    actuals = []
    with open(log_file, 'w') as log:
        log.write("===== LOOCV Detailed Calculations =====\n\n")
        log.write(f"k: {k}, Distance Metric: {distance_metric.__name__}\n")
        log.write(f"Total Instances: {len(features)}\n\n")
        print("===== LOOCV Detailed Calculations =====")

        for i in range(len(features)):
            test_instance = features[i]
            test_label = labels[i]
            training_data = np.delete(features, i, axis=0)
            training_labels = np.delete(labels, i, axis=0)

            predicted_label, neighbors = knn_predict(test_instance, training_data, training_labels, k, distance_metric)
            predictions.append(predicted_label)
            actuals.append(test_label)

            log.write(f"--- Test Instance {i+1} ---\n")
            log.write(f"Actual Label: {test_label}, Predicted Label: {predicted_label}\n")
            log.write(f"Neighbors (distance, label):\n")
            print(f"--- Test Instance {i+1} ---")
            print(f"Actual Label: {test_label}, Predicted Label: {predicted_label}")
            print(f"Neighbors (distance, label):")
            
            for dist, lbl in neighbors:
                log.write(f"  Distance: {dist:.3f}, Label: {lbl}\n")
                print(f"  Distance: {dist:.3f}, Label: {lbl}")
            
            log.write("\n")
            print()

            if predicted_label == test_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(features)
        log.write(f"\nLOOCV Accuracy: {accuracy:.2f}\n")
        print(f"\nLOOCV Accuracy: {accuracy:.2f}")
    return accuracy, predictions, actuals

# Execution
if __name__ == "__main__":
    json_path = "Dataset/play_tennis.json"
    dataset = load_dataset_from_json(json_path)
    features, labels = encode_categorical_features(dataset)
    features = normalize_features(features)  # Normalize features for better distance calculations
    features = features.values
    labels = labels.values

    print("===== Dataset =====\n")
    print(dataset.to_string(index=False))
    with open("Output/Dataset Table/dataset_log.txt", "w") as dataset_log:
        dataset_log.write("===== Dataset =====\n\n")
        dataset_log.write(dataset.to_string(index=False))

    k_values = range(1, len(features))  # Try all k values from 1 to n-1
    best_accuracy = 0
    best_k = None
    best_distance_metric = None

    for k in k_values:
        for distance_metric in [euclidean_distance, manhattan_distance]:
            log_file = f"Output/Leave-One-Out-Cross-Validation/knn_log_k{k}_{distance_metric.__name__}.txt"
            accuracy, _, _ = loocv_knn(features, labels, k, distance_metric, log_file)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_distance_metric = distance_metric

    print(f"Best k: {best_k}, Best Distance Metric: {best_distance_metric.__name__}")
    print(f"Best LOOCV Accuracy: {best_accuracy:.2f}")
    with open("Output/best_knn_log.txt", "w") as best_log:
        best_log.write(f"Best k: {best_k}, Best Distance Metric: {best_distance_metric.__name__}\n")
        best_log.write(f"Best LOOCV Accuracy: {best_accuracy:.2f}\n")