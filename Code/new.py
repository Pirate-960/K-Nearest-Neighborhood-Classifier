import pandas as pd
import numpy as np
from collections import Counter
import json
import os

# Helper Functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    magnitude = np.linalg.norm(x1) * np.linalg.norm(x2)
    return 1 - (dot_product / magnitude) if magnitude != 0 else 1


def normalize_features(features):
    """Normalize features to a 0-1 range after ensuring all columns are numeric."""
    # Convert boolean columns to integers
    features = features.astype({col: int for col in features.select_dtypes(include=["bool"]).columns})
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
def knn_predict(test_instance, training_data, training_labels, k, distance_metric, log):
    distances = []
    for i in range(len(training_data)):
        distance = distance_metric(test_instance, training_data[i])
        distances.append((distance, training_labels[i], i + 1))
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    k_nearest_labels = [label for _, label, _ in k_nearest_neighbors]

    # Log and print the nearest neighbors
    log.write("\nNearest Neighbors for test instance:\n")
    print("\nNearest Neighbors for test instance:")
    for distance, label, idx in k_nearest_neighbors:
        neighbor_info = f"Neighbor {idx} - Distance: {distance:.4f}, Label: {label}"
        print(neighbor_info)
        log.write(neighbor_info + "\n")
    
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common


def evaluate_knn(features, labels, k, distance_metric, log_file, mode="standard"):
    correct_predictions = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    results = []

    with open(log_file, 'w') as log:
        log.write(f"===== k-NN Classification ({mode}) =====\n")
        log.write(f"k: {k}, Distance Metric: {distance_metric.__name__}\n\n")

        if mode == "loocv":
            for i in range(len(features)):
                test_instance = features[i]
                test_label = labels[i]
                training_data = np.delete(features, i, axis=0)
                training_labels = np.delete(labels, i, axis=0)
                predicted_label = knn_predict(test_instance, training_data, training_labels, k, distance_metric, log)
                correct = predicted_label == test_label
                results.append((i + 1, test_label, predicted_label, correct))
                correct_predictions += correct
        else:  # Standard evaluation using entire dataset
            for i in range(len(features)):
                test_instance = features[i]
                predicted_label = knn_predict(test_instance, features, labels, k, distance_metric, log)
                correct = predicted_label == labels[i]
                results.append((i + 1, labels[i], predicted_label, correct))
                correct_predictions += correct

        # Calculate confusion matrix components
        for _, actual, predicted, _ in results:
            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 0:
                tn += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1

        accuracy = correct_predictions / len(features)

        # Calculate Precision, Recall, and F1 Score
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Print and log detailed results
        header = "--------------------------------------------------\n"
        column_headers = "| Instance |   Actual   |   Predicted  |  Correct |\n"
        divider = "--------------------------------------------------\n"

        # Print header
        print(header + column_headers + divider, end="")
        log.write(header + column_headers + divider)

        for idx, actual, predicted, correct in results:
            actual_str = "Yes" if actual == 1 else "No"
            predicted_str = "Yes" if predicted == 1 else "No"
            correct_str = "True" if correct else "False"
            row = f"|   {idx:<6} |   {actual_str:<8} |   {predicted_str:<10} |  {correct_str:<7} |\n"
            print(row, end="")
            log.write(row)

        # Print and log footer
        print(divider)
        log.write(divider)

        print(f"Overall Accuracy: {accuracy:.2f}\n\n")
        log.write(f"Overall Accuracy: {accuracy:.2f}\n\n")

        # Print and log confusion matrix
        print("--------------------------------------------------")
        print("Confusion Matrix:")
        print("--------------------------------------------------")
        print("                   Predicted")
        print("            |   Yes    |   No     |")
        print("------------|----------|----------|")
        print(f"Actual Yes  |   {tp:<6} |   {fn:<6} |")
        print(f"Actual No   |   {fp:<6} |   {tn:<6} |")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print(f"True Positives (TP):  {tp}")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print("--------------------------------------------------")

        log.write("--------------------------------------------------\n")
        log.write("Confusion Matrix:\n")
        log.write("--------------------------------------------------\n")
        log.write("                   Predicted\n")
        log.write("            |   Yes    |   No     |\n")
        log.write("------------|----------|----------|\n")
        log.write(f"Actual Yes  |   {tp:<6} |   {fn:<6} |\n")
        log.write(f"Actual No   |   {fp:<6} |   {tn:<6} |\n")
        log.write("--------------------------------------------------\n\n")
        log.write("--------------------------------------------------\n")
        log.write(f"True Positives (TP):  {tp}\n")
        log.write(f"True Negatives (TN):  {tn}\n")
        log.write(f"False Positives (FP): {fp}\n")
        log.write(f"False Negatives (FN): {fn}\n")
        log.write(f"Precision: {precision:.2f}\n")
        log.write(f"Recall: {recall:.2f}\n")
        log.write(f"F1 Score: {f1_score:.2f}\n")
        log.write("--------------------------------------------------\n")


# Execution
if __name__ == "__main__":
    print("Choose a distance metric:")
    print("1. Euclidean Distance")
    print("2. Manhattan Distance")
    print("3. Cosine Similarity")
    metric_choice = input("Enter the number of your choice (1, 2, or 3): ")

    if metric_choice == "1":
        distance_metric = euclidean_distance
    elif metric_choice == "2":
        distance_metric = manhattan_distance
    elif metric_choice == "3":
        distance_metric = cosine_similarity
    else:
        print("Invalid choice. Defaulting to Euclidean Distance.")
        distance_metric = euclidean_distance

    k = int(input("Enter the value of k: "))

    print("Choose evaluation mode:")
    print("1. LOOCV (Leave-One-Out Cross-Validation)")
    print("2. Standard Evaluation")
    mode_choice = input("Enter the number of your choice (1 or 2): ")

    mode = "loocv" if mode_choice == "1" else "standard"

    json_path = "Dataset/play_tennis.json"
    dataset = load_dataset_from_json(json_path)

    # Print and save dataset
    dataset_output_path = "Output/Dataset.txt"
    print("\nDataset Table:")
    print(dataset)
    with open(dataset_output_path, "w") as dataset_file:
        dataset_file.write(dataset.to_string())

    features, labels = encode_categorical_features(dataset)
    features = normalize_features(features).values
    labels = labels.values

    log_file = f"Output/knn_results_{mode}.txt"
    evaluate_knn(features, labels, k, distance_metric, log_file, mode)
