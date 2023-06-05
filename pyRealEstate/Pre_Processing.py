import numpy as np
import pandas as pd 

def Pandas_Missing_Data( dataf , Percent_Missing = -1 ):
  mc = dataf.isnull().sum(axis = 0)
  pm = dataf.isnull().sum() * 100 / len(dataf)
  missing_value_df = pd.DataFrame({'column_name': dataf.columns,
                                 'missing_count' : mc,
                                 'percent_missing': pm})
  return missing_value_df[missing_value_df['percent_missing'] > Percent_Missing]


def Constant_In_Data(dataf , n_unique = 1):
  uniquecols = dataf.apply(pd.Series.nunique)

  if n_unique == 1:
    rtn = uniquecols[uniquecols == 1].index.tolist()
  else :
    rtn = uniquecols[uniquecols <=  n_unique ].index.tolist()
  return rtn


def ID_in_Data(dataf ):
  ML = len(dataf)
  uniquecols = dataf.apply(pd.Series.nunique)
  return uniquecols[uniquecols == ML].index.tolist()
