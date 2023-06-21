# pyRealEstate

pyRealEstate is a library designed for data scientists working in the real estate industry. pyRealEstate is still currently under development but is aimed at providing functions to assist in the development and evaluation of AVM's. Below are some instructions on how to get started with pyRealEstate and some helpful links to descriptions and examples of all of PyRealEstates functionality. 

## Installation

The pyRealEstate package is available on [PyPi](https://pypi.org/project/pyRealEstate). Simply run: 
```
pip install pyRealEstate
```
## AVM Evaluation Metrics
pyRealEstate can provide metrics on the evaluation of your AVM (Automated Valuation Model) such as the weighted mean sale ratio, COD (Coefficient Of Disspersion), and PRD (Price Related Differential) please visit the wiki for detailed documentation [pyRealEstate Evaluation](https://github.com/Joshua-Data-Wizard/PyRealEstate/wiki/AVM-Evaluation-Metrics).

## Data Pre Processing
In addition to the evaluation metrics for the models, pyRealEstate also offers a multitude of functions to help with the pre processing of data. Please visit the wiki for detailed documentation [pyRealEstate Pre Processing](https://github.com/Joshua-Data-Wizard/PyRealEstate/wiki/Pre-Processing).

## Time Trending for AVM's
In creating Automated Valuation Models (AVM's) for real estate, it is key to capture time trends in the market. There are typically two major approaches; one is to model the time trend and then to adjust the sales based on the time adjustment rate. The other option is to include time directly in the model. If one wishes to take the first approach this pyRealEstate is to assist in the finding of the time adjustment rates. Please visit the wiki for detailed documentation [pyRealEstate Time Trending](https://github.com/Joshua-Data-Wizard/PyRealEstate/wiki/Time-Trending).
