
---

# K-Nearest Neighbors (k-NN) Classifier from Scratch

## Introduction
This project implements a K-Nearest Neighbors (k-NN) classifier from scratch in Python. The implementation showcases the principles of instance-based learning and includes various preprocessing and evaluation techniques. The "Play Tennis" dataset is used to demonstrate the functionality.

## Objectives
- Implement a k-NN classifier using Python.
- Experiment with various distance metrics and feature preprocessing methods.
- Evaluate the classifier using Leave-One-Out Cross-Validation (LOOCV).
- Log detailed calculation steps and performance metrics for transparency.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
  - [1. Distance Metrics](#1-distance-metrics)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Evaluation Modes](#3-evaluation-modes)
- [Performance Evaluation](#performance-evaluation)
- [Challenges and Solutions](#challenges-and-solutions)
  - [1. Choosing k Value](#1-choosing-k-value)
  - [2. Distance Metric Selection](#2-distance-metric-selection)
- [Results](#results)
- [References](#references)
- [License](#license)

## Dataset
The "Play Tennis" dataset, a small, categorical dataset, is used to determine if a game of tennis will be played based on weather conditions. The features include:
- **Outlook**: Sunny, Overcast, Rain
- **Temperature**: Hot, Mild, Cool
- **Humidity**: High, Normal
- **Wind**: Weak, Strong
- **PlayTennis**: Yes, No (Target variable)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pirate-960/knn-classification-project.git
   cd knn-classification-project
   ```

2. **Install Dependencies**:
   Ensure Python is installed, then run:
   ```bash
   pip install pandas numpy
   ```

3. **Directory Setup**:
   - Place the dataset in the `Dataset` folder.
   - Ensure the scripts are in the main project directory.

## Usage
1. **Run the Classifier**:
   ```bash
   python Code_w_norm.py  # For normalized features
   python Code_wo_norm.py  # For unnormalized features
   ```

2. **Interactive Prompts**:
   - Select a distance metric: Euclidean, Manhattan, or Cosine.
   - Input a k value for the nearest neighbors.
   - Choose an evaluation mode: LOOCV or Standard.

3. **Output**:
   - Classification results, confusion matrix, and performance metrics (Accuracy, Precision, Recall, F1 Score) are displayed in the console.

## Implementation Details
### 1. Distance Metrics
Three distance metrics are supported:
- **Euclidean Distance**: Measures the straight-line distance between points.
- **Manhattan Distance**: Measures the sum of absolute differences.
- **Cosine Similarity**: Measures the cosine of the angle between vectors.

### 2. Data Preprocessing
- **Normalization**: Optional scaling of features to improve performance.
- **Encoding**: Categorical features are converted to one-hot encoded representations.

### 3. Evaluation Modes
- **Leave-One-Out Cross-Validation (LOOCV)**: Evaluates performance by using one instance as a test sample and the rest for training iteratively.
- **Standard Evaluation**: Splits data into training and testing sets.

## Performance Evaluation
The classifier is evaluated using metrics such as accuracy, precision, recall, and F1 Score. Logs include intermediate calculations for transparency.

## Challenges and Solutions
### 1. Choosing k Value
- **Challenge**: Selecting an optimal k value impacts the classifier's bias-variance tradeoff.
- **Solution**: Experimented with various k values and observed their impact on performance.

### 2. Distance Metric Selection
- **Challenge**: Different metrics yield different results depending on the data distribution.
- **Solution**: Provided flexibility to choose the metric that best fits the data.

## Results
- The model achieved an accuracy of **90%+** on the "Play Tennis" dataset using LOOCV.
- Logs and performance metrics highlight the importance of feature preprocessing and distance metric selection.

## References
- **Textbooks**:
  - Alpaydin, E. (2010). *Introduction to Machine Learning*. MIT Press.
  - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- **Documentation**:
  - [NumPy Documentation](https://numpy.org/doc/)
  - [pandas Documentation](https://pandas.pydata.org/docs/)
- **Additional Resources**:
  - Wikipedia contributors. "k-Nearest Neighbors algorithm." *Wikipedia, The Free Encyclopedia*. [Read here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---