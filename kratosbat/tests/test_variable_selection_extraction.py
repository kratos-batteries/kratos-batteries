"""
These are the unit tests for the variable_selection_extraction functions
"""

import pandas as pd
import kratosbat.DataProcess.variable_selection_extraction as vse


def test_get_data_rfe():
    """
    Test that x_train and y_train rows are the same size
    """
    test_x_train = vse.get_data_rfe()[0]
    test_y_train = vse.get_data_rfe()[2]
    assert test_x_train.shape[0] == (test_y_train.shape[0]),\
    "X_train and y_train rows are not the same size"


def test_svr_linear_rfe():
    """
    Test to see if normalized SVR dataset is a pandas core dataframe
    """
    test_x_train = vse.get_data_rfe()[0]
    test_y_train = vse.get_data_rfe()[2]
    test_result = vse.svr_linear_rfe(test_x_train, test_y_train)
    assert isinstance(test_result[1], pd.core.frame.DataFrame), \
    "Not a pandas core dataframe"
