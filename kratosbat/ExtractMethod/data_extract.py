"""
This module extracts battery material data from the materials project

The module includes function which extract data for all materials on the
materials project which are classified as a battery material. It mines the
data and exports a csv where each row is a different battery material.
"""
import os
import pandas as pd
from pymatgen import MPRester
import sys
sys.path.append("../ThirdPartyResource/")
import magpie
from magpie import MagpieServer


def get_bat_dat(mapi_key):
    """
    Takes materials project API key as input and returns dataframe of
    all battery material properies

    This function returns a dataframe of all the battery materials by cycling
    through each battery ID
    """

    # MAPI_KEY is the API key obtained from the materials project
    mpr = MPRester(mapi_key)

    def get_battery_data(self, formula_or_batt_id):
        """
        Returns batteries from a batt id or formula.

        Examples:
            get_battery("mp-300585433")
            get_battery("LiFePO4")
        """
        return mpr._make_request('/battery/%s' % formula_or_batt_id)

    # adding get_battery_data function to MPRester
    MPRester.get_battery_data = get_battery_data

    # import the crytsal system table from repository
    crystal = pd.read_csv('CrystalSystemTable.csv')

    # making a list of all the battery IDs
    all_bat_ids_list = (mpr._make_request('/battery/all_ids'))

    # making an empty dataframe to hold all of the battery data
    all_battery_dataframe = pd.DataFrame([])

    # looping through every id in the list of all battery IDs
    for batt_id in all_bat_ids_list:
        # gets all the data for one battery id and stores it in a result df
        result_bat_id = pd.DataFrame(mpr.get_battery_data(batt_id))

        # this block of code goes into the adj_pairs element of the dataframe,
        # makes it a list, and extracts something from that list
        adj_pairs = result_bat_id['adj_pairs']
        adj_pairs_list = list(adj_pairs)
        in_list = pd.DataFrame(list(adj_pairs_list[0]))

        # volume change, charge and discharge properties of the materials
        max_d_vol = pd.DataFrame(in_list['max_delta_volume'])
        formula_charge = pd.DataFrame(in_list['formula_charge'])
        formula_discharge = pd.DataFrame(in_list['formula_discharge'])
        result_bat_id['Max Delta Volume'] = max_d_vol
        result_bat_id['Charge Formula'] = formula_charge
        result_bat_id['Discharge Formula'] = formula_discharge

        # go into spacegroup column and extract the crystal system number
        spacegroup_list = list(result_bat_id['spacegroup'])
        in_list_space = pd.DataFrame(spacegroup_list)
        crystal_number = in_list_space['number'][0]

        # use crystal dataframe to see what crystal system the number
        # corresponds to and store crystal system as system
        system = crystal.iloc[crystal_number-1]['Crystal']

        # append crystal system number and lattice to results
        result_bat_id['Spacegroup Number'] = crystal_number
        result_bat_id['Crystal Lattice'] = system

        # appending the results to the final dataframe
        all_battery_dataframe = all_battery_dataframe.append(result_bat_id)

    # cleaning the dataframe
    all_battery_dataframe.rename(
        columns={'battid': 'Battery ID', 'reduced_cell_formula':
                 'Reduced Cell Formula', 'average_voltage':
                 'Average Voltage (V)', 'min_voltage': 'Min Voltage (V)',
                 'max_voltage': 'Max Voltage (V)', 'nsteps': 'Number of Steps',
                 'min_instability': 'Min Instability', 'capacity_grav':
                 'Gravimetric Capacity (mAh/g)', 'capacity_vol':
                 'Volumetric Capacity (Ah/L)', 'working_ion': 'Working Ion',
                 'min_frac': 'Min Fraction', 'max_frac': 'Max Fraction',
                 'reduced_cell_composition': 'Reduced Cell Composition',
                 'framework': 'Framework', 'adj_pairs': 'Adjacent Pairs',
                 'spacegroup': 'Spacegroup', 'energy_grav':
                 'Specific Energy (Wh/kg)', 'energy_vol':
                 'Energy Density (Wh/L)', 'numsites': 'Number of Sites',
                 'type': 'Type'}, inplace=True)

    # setting index to the battery id
    clean_battery_df = all_battery_dataframe.set_index('Battery ID')

    # exports the clean dataframe as a csv to the repository
    # this ensures it does not need to be run each time
    clean_battery_df.to_csv(path_or_buf='../Data/BatteryData.csv')

    # returns the cleaned dataframe
    return clean_battery_df


def update_check(mapi_key):
    """
    This function tests to see if BatteryData.csv is up to date.

    The function checks the length of the BatteryData.csv and compares it to
    the length of the list produced by getting all of the battery IDs. If the
    lengths are the same, it returns that the data is up to date. If the
    lengths are not the same, it recommends to run get_bat_dat() to update the
    csv file.
    """
    current_df = pd.read_csv('../Data/BatteryData.csv')
    current_len = len(current_df)

    # if __name__ == "__main__":
    mpr = MPRester(mapi_key)

    all_bat_ids_list = (mpr._make_request('/battery/all_ids'))

    matproj_len = len(all_bat_ids_list)

    if current_len == matproj_len:
        print("The Current BatteryData.csv file is up to date!")

    else:
        print("Data is not up to date. Please run get_bat_dat() \
               function to obtain new .csv file.")


def get_element_property(clean_battery_df=None):
    """
    Function returns element properties from atomic constituents.

    Here we import a API called The Materials Agnostic Platform for Informatics
    and Exploration (Magpie). This API can let us use formula of a compound
    to get its  elemental properties from statistics of atomic constituents
    attributes.

    Details are in this paper:
    Ward, L.; Agrawal, A.; Choudhary, A.; Wolverton, C. A
    General-Purpose Machine Learning Framework for Predicting Properties
    of Inorganic Materials. npj Comput. Mater. 2016, 2, No. 16028.
    """

    m = MagpieServer()
    if clean_battery_df is None:
        if os.path.exists('../Data/BatteryData.csv'):
            clean_battery_df = pd.read_csv('../Data/BatteryData.csv')
    # getting Mean and Deviation of Element Property for Charge_Formula
    charge_formula = clean_battery_df['Charge Formula']
    df_mean_charge = m.generate_attributes("oqmd-Eg", charge_formula).\
        iloc[:, 6:-7:6]
    df_dev_charge = m.generate_attributes("oqmd-Eg", charge_formula).\
        iloc[:, 8:-7:6]
    df_mean_charge.rename(
        columns={'mean_Nuber': 'Char_mean_Number',
                 'mean_MendeleevNumber': 'Char_mean_MendeleevNumber',
                 'mean_AtomicWeight': 'Char_mean_AtomicWeight',
                 'mean_MeltingT': 'Char_mean_MeltingTemp',
                 'mean_Column': 'Char_mean_Column',
                 'mean_Row': 'Char_mean_Row',
                 'mean_CovalentRadius': 'Char_mean_CovalentRadius',
                 'mean_Electronegativity': 'Char_mean_Electronegativity',
                 'mean_NsValence': 'Char_mean_NsValence',
                 'mean_NpValence': 'Char_mean_NpValence',
                 'mean_NdValence': 'Char_mean_NdValence',
                 'mean_NfValence': 'Char_mean_NfValence',
                 'mean_NValance': 'Char_mean_NValance',
                 'mean_NsUnfilled': 'Char_mean_NsUnfilled',
                 'mean_NpUnfilled': 'Char_mean_NpUnfilled',
                 'mean_NdUnfilled': 'Char_mean_NdUnfilled',
                 'mean_NfUnfilled': 'Char_mean_NfUnfilled',
                 'mean_NUnfilled': 'Char_mean_NUnfilled',
                 'mean_GSvolume_pa': 'Char_mean_GSvolume_pa',
                 'mean_GSbandgap': 'Char_mean_GSbandgap',
                 'mean_GSmagmom': 'Char_mean_GSmagmom',
                 'mean_SpaceGroupNumber': 'Char_mean_SpaceGroupNumber'})
    df_dev_charge.rename(
        columns={'dev_Nuber': 'Char_dev_Number',
                 'dev_MendeleevNumber': 'Char_dev_MendeleevNumber',
                 'dev_AtomicWeight': 'Char_dev_AtomicWeight',
                 'dev_MeltingT': 'Char_dev_MeltingTemp',
                 'dev_Column': 'Char_dev_Column',
                 'dev_Row': 'Char_dev_Row',
                 'dev_CovalentRadius': 'Char_dev_CovalentRadius',
                 'dev_Electronegativity': 'Char_dev_Electronegativity',
                 'dev_NsValence': 'Char_dev_NsValence',
                 'dev_NpValence': 'Char_dev_NpValence',
                 'dev_NdValence': 'Char_dev_NdValence',
                 'dev_NfValence': 'Char_dev_NfValence',
                 'dev_NValance': 'Char_dev_NValance',
                 'dev_NsUnfilled': 'Char_dev_NsUnfilled',
                 'dev_NpUnfilled': 'Char_dev_NpUnfilled',
                 'dev_NdUnfilled': 'Char_dev_NdUnfilled',
                 'dev_NfUnfilled': 'Char_dev_NfUnfilled',
                 'dev_NUnfilled': 'Char_dev_NUnfilled',
                 'dev_GSvolume_pa': 'Char_dev_GSvolume_pa',
                 'dev_GSbandgap': 'Char_dev_GSbandgap',
                 'dev_GSmagmom': 'Char_dev_GSmagmom',
                 'dev_SpaceGroupNumber': 'Char_dev_SpaceGroupNumber'})

    # getting Mean and Deviation of Element Property for Discharge_Formula
    discharge_formula = clean_battery_df['Discharge Formula']
    df_mean_discharge = m.generate_attributes("oqmd-Eg", discharge_formula)\
        .iloc[:, 6:-7:6]
    df_dev_discharge = m.generate_attributes("oqmd-Eg", discharge_formula)\
        .iloc[:, 8:-7:6]
    df_mean_discharge.rename(
        columns={'mean_Nuber': 'Dis_mean_Number',
                 'mean_MendeleevNumber': 'Dis_mean_MendeleevNumber',
                 'mean_AtomicWeight': 'Dis_mean_AtomicWeight',
                 'mean_MeltingT': 'Dis_mean_MeltingTemp',
                 'mean_Column': 'Dis_mean_Column',
                 'mean_Row': 'Dis_mean_Row',
                 'mean_CovalentRadius': 'Dis_mean_CovalentRadius',
                 'mean_Electronegativity': 'Dis_mean_Electronegativity',
                 'mean_NsValence': 'Dis_mean_NsValence',
                 'mean_NpValence': 'Dis_mean_NpValence',
                 'mean_NdValence': 'Dis_mean_NdValence',
                 'mean_NfValence': 'Dis_mean_NfValence',
                 'mean_NValance': 'Dis_mean_NValance',
                 'mean_NsUnfilled': 'Dis_mean_NsUnfilled',
                 'mean_NpUnfilled': 'Dis_mean_NpUnfilled',
                 'mean_NdUnfilled': 'Dis_mean_NdUnfilled',
                 'mean_NfUnfilled': 'Dis_mean_NfUnfilled',
                 'mean_NUnfilled': 'Dis_mean_NUnfilled',
                 'mean_GSvolume_pa': 'Dis_mean_GSvolume_pa',
                 'mean_GSbandgap': 'Dis_mean_GSbandgap',
                 'mean_GSmagmom': 'Dis_mean_GSmagmom',
                 'mean_SpaceGroupNumber': 'Dis_mean_SpaceGroupNumber'})
    df_dev_discharge.rename(
        columns={'dev_Nuber': 'Dis_dev_Number',
                 'dev_MendeleevNumber': 'Dis_dev_MendeleevNumber',
                 'dev_AtomicWeight': 'Dis_dev_AtomicWeight',
                 'dev_MeltingT': 'Dis_dev_MeltingTemp',
                 'dev_Column': 'Dis_dev_Column',
                 'dev_Row': 'Dis_dev_Row',
                 'dev_CovalentRadius': 'Dis_dev_CovalentRadius',
                 'dev_Electronegativity': 'Dis_dev_Electronegativity',
                 'dev_NsValence': 'Dis_dev_NsValence',
                 'dev_NpValence': 'Dis_dev_NpValence',
                 'dev_NdValence': 'Dis_dev_NdValence',
                 'dev_NfValence': 'Dis_dev_NfValence',
                 'dev_NValance': 'Dis_dev_NValance',
                 'dev_NsUnfilled': 'Dis_dev_NsUnfilled',
                 'dev_NpUnfilled': 'Dis_dev_NpUnfilled',
                 'dev_NdUnfilled': 'Dis_dev_NdUnfilled',
                 'dev_NfUnfilled': 'Dis_dev_NfUnfilled',
                 'dev_NUnfilled': 'Dis_dev_NUnfilled',
                 'dev_GSvolume_pa': 'Dis_dev_GSvolume_pa',
                 'dev_GSbandgap': 'Dis_dev_GSbandgap',
                 'dev_GSmagmom': 'Dis_dev_GSmagmom',
                 'dev_SpaceGroupNumber': 'Dis_dev_SpaceGroupNumber'})

    # use concat to merge all data in one DataFrame
    element_attributes = pd.concat(
        objs=[df_mean_charge, df_dev_charge,
              df_mean_discharge, df_dev_discharge], axis=1)
    element_attributes.to_csv(path_or_buf='../Data/ElementalProperty.csv')
    return element_attributes


def get_all_variable(clean_battery_df=None, element_attributes=None):
    """
    Function to ouput our final training data for Neural Network

    features(predictor) we are to use: 'Working Ion','Crystal Lattice',
    'Spacegroup', 'element_attribute for charge formula', 'element
    attributes for discharge fromula'
    lable to be predict: 'Gravimetric Capacity (units)',
    'Volumetric Capacity','Max Delta Volume'
    """

    if clean_battery_df is None:
        if os.path.exists('../Data/BatteryData.csv'):
            clean_battery_df = pd.read_csv('../Data/BatteryData.csv')
    if element_attributes is None:
        if os.path.exists('../Data/ElementalProperty.csv'):
            element_attributes = pd.read_csv('../Data/ElementalProperty.csv')
    # select features we need in training our model
    train_set = clean_battery_df[['Working Ion', 'Crystal Lattice',
                                  'Spacegroup Number',
                                  'Gravimetric Capacity (units)',
                                  'Volumetric Capacity', 'Max Delta Volume']]
    # concat the element attributes which represent
    # the property of charge/dis electrode in our features
    train_set.reset_index(drop=False, inplace=True)
    train_set = pd.concat(objs=[train_set, element_attributes], axis=1)
    # make a .csv file to our working directory.
    train_set.to_csv(path_or_buf='../Data/TrainingData.csv')
    return train_set
