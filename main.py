""" Prediction of Company Profit Using Linear Regression
Date: 12.10.2021
Done By: Sofien Abidi
 Interpreter: Python 3.9"""

# libraries import section
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from CSV file
file_path = 'C:/Users/Sofien/Desktop/50_Startups.csv'
companies = pd.read_csv(file_path)
X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values
sns.heatmap(companies.corr())

# Encoding feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder()
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split data between training and testing
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeling and Training
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_X, train_y)

# Model Testing
y_pred = model.predict(test_X)
print(model.coef_)
print(model.intercept_)
from sklearn.metrics import r2_score
score = r2_score(y_pred, test_y)
print('Score: ', score)