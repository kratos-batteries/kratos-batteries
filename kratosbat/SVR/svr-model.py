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
     # find best parameter of model
    svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf','sigmoid','poly'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X1, y1)
    joblib.dump(svr, 'svr_CG.pkl')        # save model
    xneed = np.linspace(0, 100, 100)[:, None]
    y_pre = svr.predict(xneed)  # visualize
    plt.scatter(X1, y1, c='k', label='data', zorder=1)
    # plt.hold(True)
    plt.plot(xneed, y_pre, c='r', label='SVR_fit')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.show()
    print(svr.best_params_)
    return

def VC_svr_model(X2,y2):
    svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf','sigmoid','poly'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X2, y2)
    joblib.dump(svr, 'svr_CV.pkl')       
    xneed = np.linspace(0, 100, 100)[:, None]
    y_pre = svr.predict(xneed)
    plt.scatter(X2, y2, c='k', label='data', zorder=1)
    # plt.hold(True)
    plt.plot(xneed, y_pre, c='r', label='SVR_fit')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.show()
    print(svr.best_params_)
    return

def MDV_svr_model(X3,y3):
  
    svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf','sigmoid','poly'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr.fit(X3, y3)
    joblib.dump(svr, 'svr_MDV.pkl')     
    xneed = np.linspace(0, 100, 100)[:, None]
    y_pre = svr.predict(xneed)
    plt.scatter(X3, y3, c='k', label='data', zorder=1)
    # plt.hold(True)
    plt.plot(xneed, y_pre, c='r', label='SVR_fit')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.show()
    print(svr.best_params_)
    return