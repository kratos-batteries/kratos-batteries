from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def svr_data():
    X1=pd.read_csv('../Data/DataForSVR/GC_data.csv')
    y1=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Gravimetric Capacity (units)']]
    X2=pd.read_csv('../Data/DataForSVR/VC_data.csv')
    y2=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Volumetric Capacity']]
    X3=pd.read_csv('../Data/DataForSVR/MDV_data.csv')
    y3=pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:,['Max Delta Volume']]
    return X1,y1,X2,y2,X3,y3

def GC_svr_model(X1,y1):
     # Choose parameters automatically
    svr = GridSearchCV(SVR(), param_grid={"kernel": ('sigmoid',), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X1, y1)
    joblib.dump(svr, 'svr_CG.pkl')        # Save model
    print(svr.best_params_)
    return

def VC_svr_model(X2,y2):
     # Choose parameters automatically
    svr = GridSearchCV(SVR(), param_grid={"kernel": ('sigmoid',), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X2, y2)
    joblib.dump(svr, 'svr_CV.pkl')        # Save model
    print(svr.best_params_)
    return

def MDV_svr_model(X3,y3):
     # Choose parameters automatically
    svr = GridSearchCV(SVR(), param_grid={"kernel": ('sigmoid',), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X3, y3)
    joblib.dump(svr, 'svr_MDV.pkl')        # Save model
    print(svr.best_params_)

    return