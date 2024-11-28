import pandas as pd
import numpy as np
from collections import Counter
import json

# Helper Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

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

            # Log and Print Detailed Prediction Information
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

            # Log and Print Detailed Calculations
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

def calculate_metrics(predictions, actuals, log_file=None):
    # Initialize confusion matrix counters
    TP, TN, FP, FN = 0, 0, 0, 0

    for actual_class, predicted_class in zip(actuals, predictions):
        if actual_class == 1 and predicted_class == 1:  # Actual Yes, Predicted Yes
            TP += 1
        elif actual_class == 0 and predicted_class == 0:  # Actual No, Predicted No
            TN += 1
        elif actual_class == 0 and predicted_class == 1:  # Actual No, Predicted Yes
            FP += 1
        elif actual_class == 1 and predicted_class == 0:  # Actual Yes, Predicted No
            FN += 1

    # Calculate metrics
    accuracy = (TP + TN) / len(predictions)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Build formatted confusion matrix
    matrix_str = (
        "--------------------------------------------------\n"
        "Confusion Matrix:\n"
        "--------------------------------------------------\n"
        "                    Predicted\n"
        "            |    Yes    |     No    |\n"
        "------------|-----------|-----------|\n"
        f"Actual Yes  |   {TP:^6}  |   {FN:^6}  |\n"
        f"Actual No   |   {FP:^6}  |   {TN:^6}  |\n"
        "--------------------------------------------------\n"
        f"True Positives (TP):  {TP}\n"
        f"True Negatives (TN):  {TN}\n"
        f"False Positives (FP): {FP}\n"
        f"False Negatives (FN): {FN}\n"
        "--------------------------------------------------\n"
        f"Overall Accuracy: {accuracy:.2f}\n"
        "--------------------------------------------------\n"
    )

    print(matrix_str)

    if log_file:
        with open(log_file, 'a') as log:
            log.write(matrix_str)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }
    return metrics

# Execution
if __name__ == "__main__":
    json_path = "Dataset/play_tennis.json"
    dataset = load_dataset_from_json(json_path)
    features, labels = encode_categorical_features(dataset)
    features = features.values
    labels = labels.values

    print("===== Dataset =====\n")
    print(dataset.to_string(index=False))
    with open("Output/dataset_log.txt", "w") as dataset_log:
        dataset_log.write("===== Dataset =====\n\n")
        dataset_log.write(dataset.to_string(index=False))

    k = 3

    distance_metric = euclidean_distance
    log_without_loocv = "Output/knn_without_loocv_log.txt"
    log_with_loocv = "Output/knn_with_loocv_log.txt"

    # Without LOOCV
    predictions, actuals = evaluate_knn(features, labels, k, distance_metric, log_without_loocv)
    metrics = calculate_metrics(predictions, actuals, log_without_loocv)

    # With LOOCV
    loocv_accuracy, loocv_predictions, loocv_actuals = loocv_knn(features, labels, k, distance_metric, log_with_loocv)
    loocv_metrics = calculate_metrics(loocv_predictions, loocv_actuals, log_with_loocv)
