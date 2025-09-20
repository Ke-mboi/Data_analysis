# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Ensure plots appear inline (for Jupyter Notebook)
%matplotlib inline

# Load the Iris dataset from sklearn
iris = load_iris(as_frame=True)
df = iris.frame  # Convert to pandas DataFrame

# Display first few rows
print("First 5 rows of dataset:")
display(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean dataset (no missing values in Iris, but example for handling)
# df = df.dropna()  # OR df.fillna(value, inplace=True)
# Summary statistics
print("\nSummary Statistics:")
display(df.describe())

# Group by species and compute mean of numerical columns
grouped = df.groupby("target").mean()
print("\nMean values per species:")
display(grouped)

# Map target numbers to species names
df["species"] = df["target"].map({i: name for i, name in enumerate(iris.target_names)})

# Example finding
print("\nObservation: On average, Setosa has the smallest petals, while Virginica has the largest.")

df_sorted = df.sort_values(by="petal length (cm)")
df_sorted["cumulative_petal"] = df_sorted["petal length (cm)"].cumsum()

#Line Chart – Simulated Time Series (Cumulative Petal Length)
plt.figure(figsize=(8,5))
plt.plot(df_sorted.index, df_sorted["cumulative_petal"], label="Cumulative Petal Length", color="blue")
plt.title("Line Chart: Cumulative Petal Length")
plt.xlabel("Index (sorted by petal length)")
plt.ylabel("Cumulative Petal Length (cm)")
plt.legend()
plt.show()

# Bar Chart – Average Petal Length by Species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", palette="viridis")
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram – Distribution of Sepal Length
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=20, color="orange", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
