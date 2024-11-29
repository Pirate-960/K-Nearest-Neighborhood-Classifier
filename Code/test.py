import pandas as pd
import json

# The Play Tennis Dataset
data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
]

# Create a DataFrame from the dataset
df = pd.DataFrame(data, columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"])

# Apply One-Hot Encoding to categorical features
df_encoded = pd.get_dummies(df, columns=["Outlook", "Temperature", "Humidity", "Wind"])

# Convert the DataFrame to JSON format for further use
df_encoded.to_json("play_tennis_data.json", orient="records", lines=True)

print(df_encoded)

import numpy as np
import json
from collections import Counter

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Manhattan distance function
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# k-NN Classifier class
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):
        # Calculate distance between the test point and all training points
        distances = []
        for i in range(len(self.X_train)):
            if self.distance_metric == 'euclidean':
                dist = euclidean_distance(x, self.X_train[i])
            else:
                dist = manhattan_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        # Sort by distance and select the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Loading the dataset (from JSON file)
with open("play_tennis_data.json", "r") as f:
    data = [json.loads(line) for line in f]

# Prepare features (X) and target labels (y)
X = []
y = []

for entry in data:
    features = [entry[key] for key in entry.keys() if key != "PlayTennis"]
    X.append(features)
    y.append(entry["PlayTennis"])

X = np.array(X)
y = np.array(y)

# Instantiate the k-NN classifier
knn = KNNClassifier(k=3, distance_metric='euclidean')

# Train the classifier
knn.fit(X, y)

# Test a new instance
test_instance = [1, 0, 0, 0, 0, 1, 0, 1]  # Example: "Sunny", "Cool", "Normal", "Weak"
prediction = knn.predict([test_instance])

print(f"Predicted class: {prediction[0]}")

def leave_one_out_cross_validation(X, y):
    correct = 0
    total = len(X)
    
    for i in range(total):
        # Leave out the i-th sample for testing
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i:i+1]

        # Train the model
        knn.fit(X_train, y_train)

        # Make a prediction and check if it's correct
        prediction = knn.predict(X_test)
        if prediction[0] == y_test[0]:
            correct += 1

    accuracy = correct / total
    print(f"Leave-One-Out Cross-Validation Accuracy: {accuracy * 100:.2f}%")

# Perform LOOCV
leave_one_out_cross_validation(X, y)

from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["Yes", "No"])
    print("Confusion Matrix:")
    print(cm)

# Predict for all instances in the dataset
predictions = knn.predict(X)

# Calculate the confusion matrix
calculate_confusion_matrix(y, predictions)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
