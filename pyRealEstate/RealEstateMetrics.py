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
    if reg.pvalues[1]  < .05 :
      rtn =  reg.params[1]
    else :
      rtn = 0 
  return rtn

def PRB_Lower (y , x ): 
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
    #if reg.pvalues[1]  < .05 :
    #  rtn =  reg.conf_int(alpha=0.05, cols=None)[1,0]
    #else :
    #  rtn = 0 
    rtn =  reg.conf_int(alpha=0.05, cols=None)[1,0]
  return rtn


def PRB_Upper (y , x ): 
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
    #if reg.pvalues[1]  < .05 :
    #  rtn =  reg.conf_int(alpha=0.05, cols=None)[1,1]
    #else :
    #  rtn = 0 
    rtn =  reg.conf_int(alpha=0.05, cols=None)[1,1]
  return rtn

def PRB_Conclusion (y , x ): 
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
    if reg.pvalues[1]  > .05 or ( reg.pvalues[1]  <= .05 and np.abs(reg.params[1]) < .05 ) :
      rtn =  'PASS'
    else :
      rtn = 'FAIL'
  return rtn

def DOR_SUMMARY_Statistics ( y , x):
  print("Weighted Mean: ", weighted_Mean_Sale_Ratio(y,x),"\n")

  if COD(y,x) <= 10:
    print("COD: ",COD(y,x),"\n")
  elif COD(y,x) <= 15 :
    print("COD: ",COD(y,x)," <- NOTE THIS IS MODERATELY HIGH","\n")
  else:
    print("COD: ",COD(y,x)," <- NOTE THIS IS ABNORMALY HIGH","\n")
  
  if PRD(y,x) < .98 or PRD(y,x) > 1.03  :
    print("PRD: ", PRD(y,x), " <- NOTE THIS IS ABNORMALLY HIGH","\n" )
  else: 
    print("PRD: ", PRD(y,x),"\n")

  print("PRB: ", PRB(y,x), " <-> ", "PRB Lower Bound: ", PRB_Lower(y,x) ," <-> ", "PRB Upper Bound: ", PRB_Upper(y,x),"<->", " PRB RESULT: ", PRB_Conclusion(y,x))
