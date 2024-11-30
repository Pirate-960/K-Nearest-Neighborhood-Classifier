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
    features = features.astype({col: int for col in features.select_dtypes(include=["bool"]).columns})
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

# ---------- check this part ----------
def save_training_data_to_json(features, labels, file_path):
    """Save training data to a JSON file."""
    training_data = {"features": features.tolist(), "labels": labels.tolist()}
    with open(file_path, 'w') as f:
        json.dump(training_data, f)


def load_training_data_from_json(file_path):
    """Load training data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No training data file found at {file_path}.")
    with open(file_path, 'r') as f:
        training_data = json.load(f)
    return np.array(training_data["features"]), np.array(training_data["labels"])


def update_training_data(new_features, new_labels, file_path):
    """Update the training data stored in a JSON file."""
    try:
        features, labels = load_training_data_from_json(file_path)
        updated_features = np.vstack([features, new_features])
        updated_labels = np.concatenate([labels, new_labels])
    except FileNotFoundError:
        updated_features = new_features
        updated_labels = new_labels
    save_training_data_to_json(updated_features, updated_labels, file_path)
# ---------- check this part ----------


# Core k-NN Implementation
def knn_predict(test_instance, training_data, training_labels, k, distance_metric):
    distances = []
    for i in range(len(training_data)):
        distance = distance_metric(test_instance, training_data[i])
        distances.append((distance, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
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
                predicted_label = knn_predict(test_instance, training_data, training_labels, k, distance_metric)
                correct = predicted_label == test_label
                results.append((i + 1, test_label, predicted_label, correct))
                correct_predictions += correct
        else:  # Standard evaluation using entire dataset
            for i in range(len(features)):
                test_instance = features[i]
                predicted_label = knn_predict(test_instance, features, labels, k, distance_metric)
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

# Classify a Single Instance
def classify_single_instance(instance_json_path, model_file_path, k, distance_metric):
    """Classify a single instance provided in JSON format."""
    with open(instance_json_path, 'r') as f:
        instance = json.load(f)

    features, _ = encode_categorical_features(pd.DataFrame([instance]))
    features = normalize_features(features).values[0]

    training_features, training_labels = load_training_data_from_json(model_file_path)
    prediction = knn_predict(features, training_features, training_labels, k, distance_metric)

    print(f"Predicted Class: {'Yes' if prediction == 1 else 'No'}")


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

    print("Choose an action:")
    print("1. Train and Save Model")
    print("2. Update Existing Model")
    print("3. Classify Single Instance")
    print("4. Evaluate Model (LOOCV or Standard)")
    action_choice = input("Enter the number of your choice (1, 2, 3, or 4): ")

    model_file = "model_training_data.json"
    json_path = "Dataset/play_tennis.json"

    if action_choice == "1":
        dataset = load_dataset_from_json(json_path)
        features, labels = encode_categorical_features(dataset)
        features = normalize_features(features).values
        labels = labels.values
        save_training_data_to_json(features, labels, model_file)
        print("Training data saved successfully.")
    elif action_choice == "2":
        dataset = load_dataset_from_json(json_path)
        features, labels = encode_categorical_features(dataset)
        features = normalize_features(features).values
        labels = labels.values
        update_training_data(features, labels, model_file)
        print("Training data updated successfully.")
    elif action_choice == "3":
        instance_json_path = "Dataset/single_instance.json"
        k = int(input("Enter the value of k: "))
        classify_single_instance(instance_json_path, model_file, k, distance_metric)
    elif action_choice == "4":
        k = int(input("Enter the value of k: "))
        mode_choice = input("Choose evaluation mode (1: LOOCV, 2: Standard): ")
        mode = "loocv" if mode_choice == "1" else "standard"
        log_file = f"Output/knn_results_{mode}.txt"

        training_features, training_labels = load_training_data_from_json(model_file)
        evaluate_knn(training_features, training_labels, k, distance_metric, log_file, mode)
    else:
        print("Invalid action.")
