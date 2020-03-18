"""
These are the unit tests for the svr-model.py functions
"""

import kratosbat.SVR.svr_model as svr
import os


def test_GC_svr_model():
    X1, y1, X2, y2, X3, y3 = svr.GC_svr_model
    """
    Test whether model will be saved in a .pkl file
    """
    svr.GC_svr_model(X1, y1)
    assert os.path.exists('svr_GC.pkl'),\
    "svr_GC.pkl is not be create!"


def test_VC_svr_model():
    X1, y1, X2, y2, X3, y3 = svr.GC_svr_model
    """
    Test whether model will be saved in a .pkl file
    """
    svr.CV_svr_model(X2, y2)
    assert os.path.exists('svr_CV.pkl'),\
    "svr_CV.pkl is not be create!"


def test_MDV_svr_model():
    X1, y1, X2, y2, X3, y3 = svr.GC_svr_model
    """
    Test whether model will be saved in a .pkl file
    """
    svr.MDV_svr_model(X3, y3)
    assert os.path.exists('svr_MDV.pkl'),\
    "svr_MDV.pkl is not be create!"