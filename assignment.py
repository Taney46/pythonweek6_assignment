from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
try:
    # Fetch the Iris dataset using ucimlrepo
    iris = fetch_ucirepo(id=53)  # ID 53 corresponds to the Iris dataset

    # Convert the dataset to pandas DataFrame
    data = iris.data.features
    data['species'] = iris.data.targets

    # Display the first few rows of the dataset
    print("\nFirst five rows of the dataset:")
    print(data.head())

    # Display dataset structure
    print("\nDataset Info:")
    print(data.info())

    # Check for missing values
    print("\nMissing values per column:")
    print(data.isnull().sum())

    # Clean the dataset (drop rows with missing values)
    data = data.dropna()
    print("\nDataset cleaned. Missing values removed.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Grouping by a categorical column and computing mean of a numerical column
# Replace 'species' and 'sepal length (cm)' with appropriate column names
grouped_data = data.groupby('species')['sepal length (cm)'].mean()
print("\nMean of Sepal Length by Species:")
print(grouped_data)

# Task 3: Data Visualization
plt.figure(figsize=(10, 6))

# Line chart
# Replace 'species' and 'sepal length (cm)' with appropriate column names
plt.subplot(2, 2, 1)
data_sorted = data.sort_values('sepal length (cm)')
plt.plot(data_sorted['species'], data_sorted['sepal length (cm)'])
plt.title('Sepal Length Trend by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)

# Bar chart
plt.subplot(2, 2, 2)
grouped_data.plot(kind='bar', color='skyblue')
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.xticks(rotation=45)

# Histogram
plt.subplot(2, 2, 3)
data['sepal length (cm)'].plot(kind='hist', bins=15, color='orange', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')

# Scatter plot
plt.subplot(2, 2, 4)
plt.scatter(data['sepal width (cm)'], data['sepal length (cm)'], alpha=0.6, color='green')
plt.title('Sepal Width vs. Sepal Length')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')

# Show all plots
plt.tight_layout()
plt.show()

# Observations
print("\nObservations:")
print("1. Observation from the line chart.")
print("2. Observation from the bar chart.")
print("3. Observation from the histogram.")
print("4. Observation from the scatter plot.")
