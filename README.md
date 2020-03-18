# Krátos Batteries <img align="right" src="images/logo.png" width="150">
[![Build Status](https://travis-ci.org/kratos-batteries/kratos-batteries.svg?branch=master)](https://travis-ci.org/kratos-batteries/kratos-batteries)
[![Documentation Status](https://readthedocs.org/projects/kratos-batteries/badge/?version=latest)](https://kratos-batteries.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/kratos-batteries/kratos-batteries/badge.svg?branch=master)](https://coveralls.io/github/kratos-batteries/kratos-batteries?branch=master)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![HitCount](http://hits.dwyl.com/kratos-batteries/kratos-batteries.svg)](http://hits.dwyl.com/kratos-batteries/kratos-batteries)
## Package for Predicting Battery Parameters of New Electrode Materials
This package can be used to predict the change in volume as well as the volumetric and gravimetric capacities of new electrode materials. To do so, this package extracts the entirety of the materials project (https://materialsproject.org/) database of electrode materials and with some data manipulation is able to create a Neural Network (NN). This NN is trained to help predict the aforementioned properties of new battery materials.

### Installation
```

```
### Software Dependencies
- Python 3 (>=3.5)
  - See environment.yml for environment
Notebooks are provided in the `Examples` directory. You will require a `jupyter notebook`. 
## Organization of Repository
```
Examples/
  BatteryData.csv
  CrystalSystemsTable.csv
  NewTraingingData_MinMaxScalar.csv
  NewTraingingData_StandardScalar.csv
  README.md
  data_extract.py
  example_runtrhough.ipynb
  feature_process_SVR.ipynb
  Integrating PCA.ipynb
  nn.py
  PCA.ipynb
  SecondPassNN.ipynb
  Userdata.ipynb
doc/
  Design/
    Flow-Charts.pdf
    usecases.md
  _build/
    doctrees/
      environment.pickle
      index.doctree
    html/
      _sources/
        index.rst.txt
      _static/
        alabaster.css
        basic.css
        custom.css
        doctools.js
        documentation_options.js
        file.png
        jquery-3.4.1.js
        jquery.js
        language_data.js
        minus.png
        plus.png
        pygments.css
        searchtools.js
        underscore-1.3.1.js
        underscore.js
      .buildinfo
      genindex.html
      index.html
      objects.inv
      py-modindex.html
      search.html
      searchindex.js
  Makefile
  conf.py
  index.rst
  make.bat
images/
  SVR_CV.png
  SVR_GC.png
  SVR.MDV.png
  logo.png
kratosbat/
  Data/
    DataForSVR/
      GC_PCA.csv
      GC_data.csv
      MDV_PCA.csv
      MDV_data.csv
      VC_PCA.csv
      VC_data.csv
    BatteryData.csv
    CrystalSystemsTable.csv
    ElementalProperty.csv
    NEWTrainingData_MinMaxScaler.csv
    NEWTrainingData_StandardScaler.csv
    TrainingData.csv
  DataProcess/
    PCA.py
    magpie.py
    userdata.py
    variable_selection_extraction.py
  ExtractMethod
    data_extract.py
    elemetal_property_ext.py
    get_all_trainingdata.py
  NN/
    NEWTrainingData_MinMaxScaler.csv
    NEWTrainingData_StandardScaler.csv
    generate_inputs.py
    magpie.py
    nn.py
  SVR/
    cross_validation.py
    svr_CV.pkl
    svr_GC.pkl
    svr_MDV.pkl
    svr_model.py
  ThirdPartyResource/
    magpie.py
  tests/
    README.md
    test_cross_validation
    test_nn.py
    test_svr_model.py
    test_variable_selection_extraction
  __init__.py
  __main__.py
paper/
  Elemental Properties.pdf
  Machine Learning the Voltage of Electrode Materials.pdf
  Ward,Magpie.pdf
  supplement22.pdf
.coveragerc
.gitignore
.travis.yml
LICENSE
README.md
environment.yml
setup.py
```
## Neural Network - PyTorch
The neural network created to predict gravimetric capacity, volumetric capacity, and max volume change for battery electrodes was created using PyTorch. The network was created through a sequential model, with two hidden layers. The model was then put through 1000 epochs to reduce mean squared error before a final model was produced to predict the quantities of interest.

<img align="right" src="images/NN_lossfunction.png" width="150">
<img align="left" src="images/NeuralNetworkFlow.png" width="150">
## Usage


## Other components of README
