import numpy as np
import pandas as pd 
import statsmodels.api as sm


def weighted_Mean_Sale_Ratio ( y, x ):
  return np.mean(x) / np.mean(y)

def COD( y, x):
  ratio = x / y
  med =  np.median(ratio)
  dev = np.sum(np.abs(ratio - med))
  avgdev=dev / len(ratio)
  cod = 100 * (avgdev / med)
  return cod


def PRD(y, x):
  ratio = x / y
  mnratio = np.mean(ratio)
  mnx = np.mean(x)
  mny = np.mean(y)
  prd = mnratio / (mnx /mny)
  return prd

  
def PRB (y , x ): 
  rtn = None  
  if len(x) <= 2:
    rtn = None
  else :
    ratio = x / y
    med =  np.median(ratio)
    avmed = x / med
    value = .5 * y + .5 * avmed
    ind = np.log(value) / np.log(2)
    dep = (ratio -med) / med
    ind2 = sm.add_constant(ind)
    reg = sm.OLS(dep, ind2).fit()
    if reg.pvalues[1]  < .1 :
      rtn =  reg.params[1]
    else :
      rtn = 0 
  return rtn
