from sklearn.svm import SVR
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

'''
This part is to establish SVR model for Gravimetric Capacity, Volumetric Capacity and Max Delta Volume
prediction for certain electrode materials. Compare to multi-lable model, three seperate single-lable model
can let us select different features for different output, which can greatly reduce the noise of other less
important features.
'''

# this function used for read data in Data directory.
def svr_data():
    X1 = pd.read_csv('kratosbat/Data/DataForSVR/GC_data.csv')
    y1 = pd.read_csv('kratosbat/Data/NEWTrainingData_StandardScaler.csv').loc[:,['Gravimetric Capacity (units)']]
    X2 = pd.read_csv('kratosbat/Data/DataForSVR/VC_data.csv')
    y2 = pd.read_csv('kratosbat/Data/NEWTrainingData_StandardScaler.csv').loc[:,['Volumetric Capacity']]
    X3 = pd.read_csv('kratosbat/Data/DataForSVR/MDV_data.csv')
    y3 = pd.read_csv('kratosbat/Data/NEWTrainingData_StandardScaler.csv').loc[:,['Max Delta Volume']]
    return X1, y1, X2, y2, X3, y3

# following three fuctions are used for estabilish three different model,
# with 'rbf' kernal function based SVR model.
def GC_svr_model(X1, y1):
    X1_train, X1_test, y1_train, y1_test =train_test_split(X1, y1, test_size=0.2, random_state=123)
    svr = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=16.0, epsilon=0.1, \
              shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr.fit(X1_train, y1_train)
    joblib.dump(svr, 'svr_GC.pkl')
    y1_pred = svr.predict(X1_test)
    print(mean_squared_error(y1_test, y1_pred))
    print(r2_score(y1_test, y1_pred))
    return

def VC_svr_model(X2, y2):
    X2_train, X2_test, y2_train, y2_test =train_test_split(X2, y2, test_size=0.2, random_state=123)
    svr = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=8.0, epsilon=0.1, \
              shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr.fit(X2_train, y2_train)
    joblib.dump(svr, 'svr_CV.pkl')
    y2_pred = svr.predict(X2_test)
    print(mean_squared_error(y2_test, y2_pred))
    print(r2_score(y2_test, y2_pred))
    return

def MDV_svr_model(X3, y3):
    X3_train, X3_test, y3_train, y3_test =train_test_split(X3, y3, test_size=0.2, random_state=123)
    svr = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=4.8, epsilon=0.1, \
              shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    svr.fit(X3_train, y3_train)
    joblib.dump(svr, 'svr_MDV.pkl')
    y3_pred = svr.predict(X3_test)
    print(mean_squared_error(y3_test, y3_pred))
    print(r2_score(y3_test, y3_pred))
    return
