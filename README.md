# Titanic Survival Prediction

This project predicts whether a Titanic passenger survived or not using a logistic regression model. The code includes data preprocessing, feature engineering, model training, and various visualizations to help understand the data and evaluate the model.

## Project Overview

The main objectives of this project are:
- **Data Preprocessing:** Handling missing values, converting categorical features into numerical variables.
- **Model Training:** Using Logistic Regression to predict survival.
- **Visualization:** Creating plots to visualize data distributions, correlations, and model performance (confusion matrix, ROC curve).

## Files

- **`code.py`**: Contains the full code for loading the data, preprocessing, training the model, and visualizing the results.
- **`tested.csv`**: The dataset file used in the project (ensure the file path in the script points to its correct location).
- **`README.md`**: This file.

## Prerequisites

Ensure you have Python 3 installed. You will also need the following Python libraries:

- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

You can install these libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

# How to Run the Project
## 1. Clone the Repository

Open your terminal or command prompt and run:

``` bash
git clone https://github.com/chandan14rta/Titanic_Survival.git
```

## 2.Navigate to the Project Directory

``` bash
cd Titanic_Survival
```
## 3. Run the Python Script

Make sure the tested.csv file is in the correct path as specified in your script. Then run:

```bash
python code.py
```
Replace code.py with the actual name of your Python file if it's different.

# Code Overview
The code performs the following steps:

**1. Importing Libraries:** Loads necessary Python libraries.
**2. Data Loading and Cleaning:** Reads the CSV file, fills missing values (for Age), and drops remaining missing values.
**3. Feature Engineering:** Converts categorical variables such as Sex and Embarked into dummy variables.
**4. Data Visualization:**
 **-** Distribution of Survivors: #### Uses a count plot.
 **-** Age Distribution by Survival Status: #### Uses a histogram with KDE.
 **-** Feature Correlation: #### Uses a heatmap.
**5. Model Training:** Splits the dataset into training and testing sets, fits a logistic regression model.
**6. Model Evaluation:**
 **-** Displays the model accuracy.
 **-** Plots a confusion matrix.
 **-** Generates an ROC curve.
 **-** Prints a classification report.
