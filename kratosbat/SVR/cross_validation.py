from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def svr_data():
    X1=pd.read_csv('../Data/DataForSVR/GC_data.csv')
    y1=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Gravimetric Capacity (units)']]
    X2=pd.read_csv('../Data/DataForSVR/VC_data.csv')
    y2=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Volumetric Capacity']]
    X3=pd.read_csv('../Data/DataForSVR/MDV_data.csv')
    y3=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Max Delta Volume']]
    return X1,y1,X2,y2,X3,y3


def GC_kernal_select(X1,y1):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X1, y1)
    kernal=['linear','poly', 'rbf','sigmoid', 'precomputed']

    svr=SVR(kernel='sigmoid', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    mse=[]
    for item in kernal:
        svr=SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X1, y1):
        #print("TRAIN:", train_index, "TEST:", test_index)

          X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
          y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]

          svr.fit(X1_train, y1_train)
          y1_pred = svr.predict(X1_test)

    mse.append(pd.Series(mean_squared_error(y1_test, y1_pred)).mean())
    print(mse)
    return


def VC_kernal_select(X2,y2):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X2, y2)
    kernal=['linear','poly', 'rbf','sigmoid', 'precomputed']

    svr=SVR(kernel='sigmoid', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    mse=[]
    for item in kernal:
        svr=SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X2, y2):
        #print("TRAIN:", train_index, "TEST:", test_index)
        
          X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]
          y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]

          svr.fit(X2_train, y2_train)
          y2_pred = svr.predict(X2_test)

    mse.append(pd.Series(mean_squared_error(y2_test, y2_pred)).mean())
    print(mse)
    return


def VC_kernal_select(X3,y3):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X3, y3)
    kernal=['linear','poly', 'rbf','sigmoid', 'precomputed']

    svr=SVR(kernel='sigmoid', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    mse=[]
    for item in kernal:
        svr=SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X3, y3):
        #print("TRAIN:", train_index, "TEST:", test_index)
        
          X3_train, X3_test = X3.iloc[train_index], X3.iloc[test_index]
          y3_train, y3_test = y3.iloc[train_index], y3.iloc[test_index]

          svr.fit(X3_train, y3_train)
          y3_pred = svr.predict(X3_test)

    mse.append(pd.Series(mean_squared_error(y3_test, y3_pred)).mean())
    print(mse)
    return