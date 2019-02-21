import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd


data=pd.read_csv('training.csv')
print data.head()

y = data.Image
X = data.drop(['Image'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

