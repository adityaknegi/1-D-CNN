
import numpy as np
import pandas as pd
from sklearn import preprocessing


def liver():
    Train = pd.read_csv('dataset/diabetes.csv')
    Train.head()
    
    if(pd.isnull(Train).any().any()):
        print(" Error :missing value in dataset")
        return



    X =Train[Train.columns[:-1]]
    y=Train[Train.columns[-1]]

    x = X.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)
    X = x_scaled
    y=y.values
    return X,y


def ionosphere():
    Train = pd.read_csv('dataset/ionosphere.data',header =None)

    if(pd.isnull(Train).any().any()):
        
        print(" Error :missing value in dataset")
        
        return

    X =Train[Train.columns[:-1]]
    y=Train[Train.columns[-1]]
    y.replace({'g':0,'b':1},inplace=True)


    x = X.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)
    X = x_scaled
    y=y.values
    return X,y


def wbc():
    Train = pd.read_csv('dataset/breast-cancer-wisconsin.data',header=None)
    Train.drop(labels = Train.columns[0],axis=1,inplace=True)
    #pd.isnull(Train).any()
    col_no=np.argmax(np.ravel((Train=='?').any()))
    # 16 attributes in col 6 with ? attributes
    # all index with ? value 
    index =Train[Train[Train.columns[col_no]]=='?'].index

    Train.drop(labels=index,axis=0,inplace=True)
    Train.reset_index(drop=True,inplace=True)


    X =Train[Train.columns[:-1]]
    y=Train[Train.columns[-1]]//2-1
    # convert to string value to numeric value
    X[X.columns[5]] = pd.to_numeric(X[X.columns[5]], errors='coerce')

    x = X.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)
    X = x_scaled
    y=y.values
    return X,y

def LSVT_voice_rehabilitation():
	X = pd.read_excel('dataset/LSVT_voice_rehabilitation.xlsx',sheet_name=0)
	y = pd.read_excel('dataset/LSVT_voice_rehabilitation.xlsx',sheet_name=1)
	y=y-1

	x = X.values #returns a numpy array
	# covert to rnge 0 to 1
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

	x_scaled = min_max_scaler.fit_transform(x)
	X = x_scaled
	y=y.values
	return X,y


