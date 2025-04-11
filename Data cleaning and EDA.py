import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Get basic statistics
print(df.describe())
# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Convert categorical variables into numerical format
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print(df.info())

# Check data after cleaning
import seaborn as sns
import matplotlib.pyplot as plt

# Plot survival rate by gender
sns.barplot(x="Sex_male", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.show()
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution of Titanic Passengers")
plt.show()
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()
import numpy as np

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()