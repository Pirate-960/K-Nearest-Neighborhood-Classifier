import pandas as pd
import numpy as np
from collections import Counter
import json
import os

# Helper Functions for Distance Calculations and Feature Normalization

def euclidean_distance(x1, x2):
    """
    Calculate Euclidean distance between two vectors.
    Euclidean distance is the straight-line distance between two points in n-dimensional space.
    
    Args:
        x1 (numpy.array): First vector
        x2 (numpy.array): Second vector
    
    Returns:
        float: Euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Calculate Manhattan (City Block) distance between two vectors.
    Manhattan distance measures the sum of absolute differences between coordinates.
    
    Args:
        x1 (numpy.array): First vector
        x2 (numpy.array): Second vector
    
    Returns:
        float: Manhattan distance between x1 and x2
    """
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1, x2):
    """
    Calculate cosine similarity between two vectors.
    Measures the cosine of the angle between two non-zero vectors.
    Returns a distance metric (1 - similarity) to be consistent with other distance metrics.
    
    Args:
        x1 (numpy.array): First vector
        x2 (numpy.array): Second vector
    
    Returns:
        float: Cosine distance (1 - cosine similarity)
    """
    dot_product = np.dot(x1, x2)
    magnitude = np.linalg.norm(x1) * np.linalg.norm(x2)
    return 1 - (dot_product / magnitude) if magnitude != 0 else 1


def normalize_features(features):
    """
    Normalize features to a 0-1 range using min-max normalization.
    Converts boolean columns to integers and scales all numeric features.
    
    Args:
        features (pandas.DataFrame): Input feature dataframe
    
    Returns:
        pandas.DataFrame: Normalized features with all values between 0 and 1
    """
    # Convert boolean columns to integers
    features = features.astype({col: int for col in features.select_dtypes(include=["bool"]).columns})
    # Perform min-max normalization: (x - min(x)) / (max(x) - min(x))
    return (features - features.min()) / (features.max() - features.min())


# Data Preparation and Management Functions

def load_dataset_from_json(file_path):
    """
    Load a dataset from a JSON file and convert it to a pandas DataFrame.
    
    Args:
        file_path (str): Path to the JSON file containing the dataset
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def print_and_save_dataset(dataset, output_path):
    """
    Print dataset to console and save it to a file in markdown format.
    Creates the output directory if it doesn't exist.
    
    Args:
        dataset (pandas.DataFrame): Dataset to print and save
        output_path (str): File path to save the dataset
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Format the dataset for clear console display
    formatted_dataset = dataset.to_markdown(index=False, tablefmt="grid")

    # Print dataset to console
    print("\n=== Dataset Table ===")
    print(formatted_dataset)

    # Save dataset to the output file
    with open(output_path, "w") as dataset_file:
        dataset_file.write("# Dataset Table\n")
        dataset_file.write(formatted_dataset)
        dataset_file.write("\n")


def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding and map target variable.
    
    Args:
        df (pandas.DataFrame): Input dataframe with categorical features
    
    Returns:
        tuple: 
            - features_encoded (pandas.DataFrame): One-hot encoded features
            - target_encoded (pandas.Series): Binary encoded target variable
    """
    # Separate features and target
    features = df.drop(columns=['PlayTennis'])
    target = df['PlayTennis']
    
    # One-hot encode categorical features, dropping first column to avoid multicollinearity
    features_encoded = pd.get_dummies(features, drop_first=True)
    
    # Map target variable to binary (Yes: 1, No: 0)
    target_encoded = target.map({'Yes': 1, 'No': 0})
    
    return features_encoded, target_encoded


# Core k-NN (k-Nearest Neighbors) Implementation

def knn_predict(test_instance, training_data, training_labels, k, distance_metric, calculations_log):
    """
    Perform k-Nearest Neighbors prediction for a single test instance.
    
    Args:
        test_instance (numpy.array): Single test data point
        training_data (numpy.array): Training dataset
        training_labels (numpy.array): Labels for training dataset
        k (int): Number of nearest neighbors to consider
        distance_metric (function): Distance calculation function
        calculations_log (file): File to log distance calculations
    
    Returns:
        int: Predicted label based on k-nearest neighbors
    """
    distances = []
    calculation_title = f"\n=== Distance Calculations for Test Instance ==="
    print(calculation_title)
    calculations_log.write(calculation_title + "\n")
    print("-" * 50)
    calculations_log.write("-" * 50 + "\n")

    # Compute distances to all training instances
    for i, training_instance in enumerate(training_data):
        distance = distance_metric(test_instance, training_instance)
        distances.append((distance, training_labels[i], i + 1))
        calculation_detail = f"Neighbor {i + 1}: Distance = {distance:.4f} -> Label = {training_labels[i]}"
        print(calculation_detail)
        calculations_log.write(calculation_detail + "\n")

    # Sort distances and select k-nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    neighbor_title = "\n--- Selected Nearest Neighbors (Top k) ---"
    print(neighbor_title)
    calculations_log.write(neighbor_title + "\n")
    print("-" * 50)
    calculations_log.write("-" * 50 + "\n")

    for rank, (distance, label, idx) in enumerate(k_nearest_neighbors, 1):
        neighbor_info = f"Rank {rank}: Neighbor {idx} - Distance = {distance:.4f} -> Label = {label}"
        print(neighbor_info)
        calculations_log.write(neighbor_info + "\n")

    # Determine the most common label among k-nearest neighbors using majority voting
    k_nearest_labels = [label for _, label, _ in k_nearest_neighbors]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common


def evaluate_knn(features, labels, k, distance_metric, results_file, calculations_file, mode="standard"):
    """
    Evaluate k-Nearest Neighbors classification using selected mode and metric.
    
    Args:
        features (numpy.array): Normalized feature data
        labels (numpy.array): Corresponding labels
        k (int): Number of nearest neighbors
        distance_metric (function): Distance calculation function
        results_file (str): Path to save classification results
        calculations_file (str): Path to save distance calculations
        mode (str, optional): Evaluation mode ('standard' or 'loocv'). Defaults to 'standard'.
    """
    correct_predictions = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    results = []

    with open(results_file, 'w') as results_log, open(calculations_file, 'w') as calculations_log:
        results_log.write(f"===== k-NN Classification Results ({mode}) =====\n")
        results_log.write(f"k: {k}, Distance Metric: {distance_metric.__name__}\n\n")

        # Two different evaluation modes: Leave-One-Out Cross-Validation (LOOCV) and Standard
        if mode == "loocv":
            # In LOOCV, each instance is used as a test point once
            for i in range(len(features)):
                test_instance = features[i]
                test_label = labels[i]
                # Remove current instance from training data
                training_data = np.delete(features, i, axis=0)
                training_labels = np.delete(labels, i, axis=0)
                # Predict label and check correctness
                predicted_label = knn_predict(test_instance, training_data, training_labels, k, distance_metric, calculations_log)
                correct = predicted_label == test_label
                results.append((i + 1, test_label, predicted_label, correct))
                correct_predictions += correct
        else:  # Standard evaluation using entire dataset
            # In standard mode, each instance is classified using the entire dataset
            for i in range(len(features)):
                test_instance = features[i]
                predicted_label = knn_predict(test_instance, features, labels, k, distance_metric, calculations_log)
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

        # Calculate accuracy
        accuracy = correct_predictions / len(features)

        # Calculate Precision, Recall, and F1 Score
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Detailed logging and console output of results
        header = "--------------------------------------------------\n"
        column_headers = "| Instance |   Actual   |   Predicted  |  Correct |\n"
        divider = "--------------------------------------------------\n"

        # Print and log detailed per-instance results
        print(header + column_headers + divider, end="")
        results_log.write(header + column_headers + divider)

        for idx, actual, predicted, correct in results:
            actual_str = "Yes" if actual == 1 else "No"
            predicted_str = "Yes" if predicted == 1 else "No"
            correct_str = "True" if correct else "False"
            row = f"|   {idx:<6} |   {actual_str:<8} |   {predicted_str:<10} |  {correct_str:<7} |\n"
            print(row, end="")
            results_log.write(row)

        # Print and log results summary and confusion matrix
        print(divider)
        results_log.write(divider)

        print(f"Overall Accuracy: {accuracy:.2f}\n\n")
        results_log.write(f"Overall Accuracy: {accuracy:.2f}\n\n")

        # Detailed console and file output of performance metrics
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

        # Log the same information to results file
        results_log.write("--------------------------------------------------\n")
        results_log.write("Confusion Matrix:\n")
        results_log.write("--------------------------------------------------\n")
        results_log.write("                   Predicted\n")
        results_log.write("            |   Yes    |   No     |\n")
        results_log.write("------------|----------|----------|\n")
        results_log.write(f"Actual Yes  |   {tp:<6} |   {fn:<6} |\n")
        results_log.write(f"Actual No   |   {fp:<6} |   {tn:<6} |\n")
        results_log.write("--------------------------------------------------\n\n")
        results_log.write("--------------------------------------------------\n")
        results_log.write(f"True Positives (TP):  {tp}\n")
        results_log.write(f"True Negatives (TN):  {tn}\n")
        results_log.write(f"False Positives (FP): {fp}\n")
        results_log.write(f"False Negatives (FN): {fn}\n")
        results_log.write(f"Precision: {precision:.2f}\n")
        results_log.write(f"Recall: {recall:.2f}\n")
        results_log.write(f"F1 Score: {f1_score:.2f}\n")
        results_log.write("--------------------------------------------------\n")


# Main Execution Block
if __name__ == "__main__":
    # User interface for selecting distance metric
    print("Choose a distance metric:")
    print("1. Euclidean Distance")
    print("2. Manhattan Distance")
    print("3. Cosine Similarity")
    metric_choice = input("Enter the number of your choice (1, 2, or 3): ")

    # Map user choice to corresponding distance metric
    if metric_choice == "1":
        distance_metric = euclidean_distance
    elif metric_choice == "2":
        distance_metric = manhattan_distance
    elif metric_choice == "3":
        distance_metric = cosine_similarity
    else:
        print("Invalid choice. Defaulting to Euclidean Distance.")
        distance_metric = euclidean_distance

    # Get k value for k-NN from user
    k = int(input("Enter the value of k: "))

    # User interface for selecting evaluation mode
    print("Choose evaluation mode:")
    print("1. LOOCV (Leave-One-Out Cross-Validation)")
    print("2. Standard Evaluation")
    mode_choice = input("Enter the number of your choice (1 or 2): ")

    # Set mode based on user choice
    mode = "loocv" if mode_choice == "1" else "standard"

    # Load dataset from JSON file
    json_path = "Dataset/play_tennis.json"
    dataset = load_dataset_from_json(json_path)

    # Save and display dataset
    dataset_output_path = "Output/Dataset.txt"
    print_and_save_dataset(dataset, dataset_output_path)

    # Encode categorical features and normalize them
    features, labels = encode_categorical_features(dataset)
    features = normalize_features(features).values  # Convert to NumPy array for computation
    labels = labels.values  # Convert to NumPy array for computation

    # Set up output file paths for results and calculations
    results_file = f"Output/knn_results_{mode}.txt"
    calculations_file = f"Output/knn_calculations_{mode}.txt"

    # Evaluate k-NN classifier with selected parameters
    evaluate_knn(features, labels, k, distance_metric, results_file, calculations_file, mode)