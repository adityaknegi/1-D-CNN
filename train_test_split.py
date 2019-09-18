import numpy as np 
from sklearn.model_selection import train_test_split


def split(X,y):
	X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.20, random_state=1994)
	return (np.array(X_train).T,np.array(X_test).T, np.array(y_train).reshape(1,-1), np.array(y_test).reshape(1,-1))