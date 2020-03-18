# Krátos Batteries <img align="right" src="images/logo.png" width="150">
[![Build Status](https://travis-ci.org/kratos-batteries/kratos-batteries.svg?branch=master)](https://travis-ci.org/kratos-batteries/kratos-batteries)
[![Coverage Status](https://coveralls.io/repos/github/kratos-batteries/kratos-batteries/badge.svg?branch=master)](https://coveralls.io/github/kratos-batteries/kratos-batteries?branch=master)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![HitCount](http://hits.dwyl.com/kratos-batteries/kratos-batteries.svg)](http://hits.dwyl.com/kratos-batteries/kratos-batteries)
## Package for Predicting Battery Parameters of New Electrode Materials
This package can be used to predict the change in volume as well as the volumetric and gravimetric capacities of new electrode materials. To do so, this package extracts the entirety of the materials project (https://materialsproject.org/) database of electrode materials and with some data manipulation is able to create a Neural Network (NN). This NN is trained to help predict the aforementioned properties of new battery materials.

### Installation
```
install code here (need a package name)
```
### Software Dependencies
- Python 3 (>=3.6)
  - See environment.yml for environment
Notebooks are provided in the `Examples` directory. You will require a `jupyter notebook`. 
## Organization of Repository
```
doc/
  Flow-Charts.pdf
  usecases.md
examples/
  BatteryData.csv
  CrystalSystemsTable.csv
  README.md
  data_extract.py
  example_runtrhough.ipynb
  feature_process_SVR.ipynb
images/
  logo.png
kratosbat/
  Data/
    DataForSVR/
      GC_CPA.csv
      GC_data.csv
      MDV_CPA.csv
      MDV_data.csv
      VC_CPA.csv
      VC_data.csv
    BatteryData.csv
    CrystalSystemsTable.csv
    ElementalProperty.csv
    NEWTrainingData_MinMaxScaler.csv
    NEWTrainingData_StandardScaler.csv
    TrainingData.csv
  DataProcess/
    PCA.py
    README.md
    test_PCA.py
    test_variable_selection_extraction.py
    variable_selection_extraction.py
  ExtractMethod
    data_extract.py
    elemetal_property_ext.py
    get_all_trainingdata.py
  NN/
    NEWTrainingData_MinMaxScaler.csv
    NEWTrainingData_StandardScaler.csv
    NN.ipynb
    SecondPass.ipynb
    TrainingData.csv
    nn.py
    test_nn.py
  SVR/
    cross_validation.py
    svr-model.py
    svr_script.ipynb
  ThirdPartyResource/
    magpie.py
  tests/
    test_core.py
  core.py
  __main__.py
paper/
  Elemental Properties.pdf
  Machine Learning the Voltage of Electrode Materials.pdf
  Ward,Magpie.pdf
  supplement22.pdf
.gitignore
.travis.yml
LICENSE
README.md
environment.yml
setup.py
```
## Neural Network - PyTorch
The neural network created to predict gravimetric and volumetric capacity, as well as max volume change, for battery electrodes was created using PyTorch. The network was created through a sequential model, with two hidden layers. The model was then put through 1000 epochs to reduce mean squared error before a final model was produced to predict the quantities of interest.
## Usage


## Other components of README
