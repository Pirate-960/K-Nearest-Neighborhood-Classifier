import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Dataset/play_tennis.csv")

# Display the first few rows of the dataset
print(data.head())

# Display the shape of the dataset  
print(data.shape)

# Display the summary statistics of the dataset
print(data.describe())
