import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd


data = pd.read_csv('training.csv')
print data.head()

#y = data.Image
#X = data.drop(['Image'], axis=1)

X_train, X_test = train_test_split(data,test_size=0.2)
print("\nX_train:\n")
print X_train['left_eye_inner_corner_x']
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

#np.savetxt("split_train.csv", X_train, delimiter=",", fmt='%s')
#np.savetxt("split_test.csv", X_test, delimiter=",")

