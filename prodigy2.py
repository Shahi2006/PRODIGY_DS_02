# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Display basic information
print("Dataset Overview:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 1: Data Cleaning
# Fill missing Age values with the median
data.fillna({'Age':data['Age'].median()}, inplace=True)

# Drop the Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True)

# Fill missing Embarked values with the most frequent category
data.fillna({'Embarked':data['Embarked'].mode()[0]}, inplace=True)

# Convert categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Check for duplicates and remove them
data.drop_duplicates(inplace=True)

# Step 2: Exploratory Data Analysis (EDA)

# Univariate Analysis: Distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Univariate Analysis: Survival Rate
plt.figure(figsize=(8, 6))
data['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis: Survival by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=data,)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()

# Bivariate Analysis: Survival by Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class (1 = First, 2 = Second, 3 = Third)')
plt.ylabel('Survival Rate')
plt.show()

# Bivariate Analysis: Survival by Age
plt.figure(figsize=(10, 6))
sns.kdeplot(data[data['Survived'] == 1]['Age'], label='Survived', fill=True, color='green')
sns.kdeplot(data[data['Survived'] == 0]['Age'], label='Did Not Survive', fill=True, color='red')
plt.title('Survival by Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Multivariate Analysis: Survival by Gender and Class
plt.figure(figsize=(10, 6))
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=data, kind='bar')
plt.title('Survival Rate by Gender and Class')
plt.xlabel('Passenger Class (1 = First, 2 = Second, 3 = Third)')
plt.ylabel('Survival Rate')
plt.show()


# Summary of findings
print("\nSummary of Findings:")
print("1. Women had a higher survival rate than men.")
print("2. Passengers in 1st class had the highest survival rate, followed by 2nd and 3rd class.")
print("3. Younger passengers were more likely to survive.")

