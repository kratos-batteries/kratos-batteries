def get_all_variable(clean_battery_df=None,element_attributes=None):

     #features(predictor) we are to use: 'Working Ion','Crystal Lattice','Spacegroup', 'element_attribute for charge formula', 'element attributes for discharge fromula'
     # lable to be predict: 'Gravimetric Capacity (units)','Volumetric Capacity','Max Delta Volume'
     import os
     import pandas as pd
     if clean_battery_df == None:
         if os.path.exists('./Data/BatteryData.csv'):
             clean_battery_df=pd.read_csv('./Data/BatteryData.csv')
     if element_attributes == None:
         if os.path.exists('./Data/ElementalProperty.csv'):
             element_attributes=pd.read_csv('./Data/ElementalProperty.csv')
     # select features we need in training our model
     train_set=clean_battery_df[['Working Ion','Crystal Lattice','Spacegroup Number','Gravimetric Capacity (units)','Volumetric Capacity','Max Delta Volume']]
     # concat the element attributes which represent the property of charge/dis electrode in our features
     train_set.reset_index(drop=False, inplace=True)
     train_set=pd.concat(objs=[train_set,element_attributes],axis=1)
     # make a .csv file to our working directory.
     train_set.to_csv(path_or_buf='./Data/TrainingData.csv')
     return train_set