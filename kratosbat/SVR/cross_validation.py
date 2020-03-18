from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
this file is to use 10 fold cross-validation to select best kernal function 
that give best performence in SVR model.

after establishing our SVR model, we will use cross-validation to evaluate our 
model by get average MSE, RSS, R^2

'''

# this functon is used to read data set into Dataframe.


def svr_data():
    X1 = pd.read_csv('../Data/DataForSVR/GC_PCA.csv')
    y1 = pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:, ['Gravimetric Capacity (units)']]
    X2 = pd.read_csv('../Data/DataForSVR/VC_PCA.csv')
    y2 = pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:, ['Volumetric Capacity']]
    X3 = pd.read_csv('../Data/DataForSVR/MDV_PCA.csv')
    y3 = pd.read_csv('../Data/NEWTrainingData_StandardScaler.csv').loc[:, ['Max Delta Volume']]
    return X1, y1, X2, y2, X3, y3


# following three functions are in same figure: use 10 fold cross-validation select best 
# kernal in a list(ploy, rbf, sigmoid),and show the outcome as MSE for each kernal model
def GC_kernal_select(X1, y1):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X1, y1)
    kernal = ['poly', 'rbf', 'sigmoid']
    mse = []
    for item in kernal:
        svr = SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                  shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X1, y1):
            X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
            y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]
            svr.fit(X1_train, y1_train.values.ravel())
            y1_pred = svr.predict(X1_test)
            mse.append(pd.Series(mean_squared_error(y1_test, y1_pred)).mean())
    print(mse)
    return mse


def VC_kernal_select(X2, y2):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X2, y2)
    kernal = ['poly', 'rbf', 'sigmoid']
    mse = []
    for item in kernal:
        svr = SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                  shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X2, y2):
            X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]
            y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]
            svr.fit(X2_train, y2_train.values.ravel())
            y2_pred = svr.predict(X2_test)

            mse.append(pd.Series(mean_squared_error(y2_test, y2_pred)).mean())
    print(mse)
    return mse


def VC_kernal_select(X3, y3):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X3, y3)
    kernal = ['poly', 'rbf', 'sigmoid']
    mse = []
    for item in kernal:
        svr = SVR(kernel=item, degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, \
                  shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        for train_index, test_index in kf.split(X3, y3):
            X3_train, X3_test = X3.iloc[train_index], X3.iloc[test_index]
            y3_train, y3_test = y3.iloc[train_index], y3.iloc[test_index]
            svr.fit(X3_train, y3_train.values.ravel())
            y3_pred = svr.predict(X3_test)

            mse.append(pd.Series(mean_squared_error(y3_test, y3_pred)).mean())
    print(mse)
    return mse


# this function is working on test and evaluate the performance of our estabilshed model
# get from svr-model.py
def SVR_model_validation(X1, y1, X2, y2, X3, y3):
    svr_cg = joblib.load('svr_GC.pkl')
    svr_cv = joblib.load('svr_CV.pkl')
    svr_mdv = joblib.load('svr_MDV.pkl')
     
    # evaluate model of GC
    kf1 = KFold(n_splits=10, shuffle=True)
    kf1.get_n_splits(X1, y1)   
    list_mse1 = []
    list_R21 = []
    for train_index, test_index in kf1.split(X1, y1):
        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]
        y1_train, y1_test = y1.iloc[train_index], y1.iloc[test_index]
        y1_pred = svr_cg.predict(X1_test)
        list_mse1.append(mean_squared_error(y1_test, y1_pred))
        list_R21.append(r2_score(y1_test, y1_pred))
    print(pd.Series(list_mse1).mean())
    print(pd.Series(list_R21).mean())

    # evaluate model of VC
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X2, y2)   
    list_mse2 = []
    list_R22 = []
    for train_index, test_index in kf.split(X2, y2):
        X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]
        y2_train, y2_test = y2.iloc[train_index], y2.iloc[test_index]
        y2_pred = svr_cv.predict(X2_test)
        list_mse2.append(mean_squared_error(y2_test, y2_pred))
        list_R22.append(r2_score(y2_test, y2_pred))
    print(pd.Series(list_mse2).mean())
    print(pd.Series(list_R22).mean())

    # evaluate model of MDV
    kf3 = KFold(n_splits=10, shuffle=True)
    kf3.get_n_splits(X3, y3)   
    list_mse3 = []
    list_R23 = []
    for train_index, test_index in kf3.split(X2, y2):
        X3_train, X3_test = X3.iloc[train_index], X3.iloc[test_index]
        y3_train, y3_test = y3.iloc[train_index], y3.iloc[test_index]
        y3_pred = svr_mdv.predict(X3_test)
        list_mse3.append(mean_squared_error(y3_test, y3_pred))
        list_R23.append(r2_score(y3_test, y3_pred))
    print(pd.Series(list_mse3).mean())
    print(pd.Series(list_R23).mean())
    return list_R21, list_R22, list_R23, list_mse1, list_mse2, list_mse3
