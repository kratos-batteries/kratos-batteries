"""
These are the unit tests for the variable_selection_extraction functions
"""

import kratosbat.SVR.cross_validation as cv


def test_GC_kernal_select():
    X1, y1, X2, y2, X3, y3 = cv.svr_data()
    """
    Test whether model will be saved in a .pkl file
    """
    mse = cv.GC_kernal_select(X1, y1)
    assert len(mse) == 10,\
        "Cross validation do not have 10 folds!"


def test_SVR_model_validation():
    X1, y1, X2, y2, X3, y3 = cv.svr_data()
    """
    Test whether model will be saved in a .pkl file
    """
    list_R21, list_R22, list_R23, list_mse1, list_mse2, list_mse3 = \
        cv.SVR_model_validation(X1, y1, X2, y2, X3, y3)

    assert len(list_R21) == len(list_mse1),\
        "Cross validation do not have 10 folds!"
