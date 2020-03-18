import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def convert(scaled):
    """Function to convert from scaled data to normal data"""
    df_output=pd.read_csv('../kratosbat/Data/TrainingData.csv')[['Gravimetric Capacity (units)','Volumetric Capacity']]
    ms = MinMaxScaler()
    ms.fit(DF_OUTPUT)
    result=MS.inverse_transform(y)
    print(result)
