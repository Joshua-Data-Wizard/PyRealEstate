# pyRealEstate

pyRealEstate is a library designed for data scientists working in the real estate industry. pyRealEstate is still currently under development but is aimed at providing functions to assist in the development and evaluation of AVM's. 

## Instructions

1. Install:

```
pip3 install pyRealEstate
```
## Tutorial on how to test AVM  Performance 

pyRealEstate can calculate the COD (Coefficient of Dispersion) and PRD (Price Related Differential) as shown below 

```
import numpy as np 
from pyRealEstate import RealEstateMetrics
appr = np.array([25500,57000,39000,90000,51000,93000,49500])
sale = np.array([75000,150000,90000,180000,90000,150000,75000])
RealEstateMetrics.COD(appr,sale)
```
