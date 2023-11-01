from typing import Optional

import numpy as np
# not used
# import pandas as pd
import statsmodels.api as sm


def weighted_Mean_Sale_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    return np.mean(x) / np.mean(y)


def COD(y: np.ndarray, x: np.ndarray) -> float:
    ratio = x / y
    med = np.median(ratio)
    dev = np.sum(np.abs(ratio - med))
    avgdev = dev / len(ratio)
    cod = 100 * (avgdev / med)
    return cod


def PRD(y: np.ndarray, x: np.ndarray) -> float:
    ratio = x / y
    mnratio = np.mean(ratio)
    mnx = np.mean(x)
    mny = np.mean(y)
    prd = mnratio / (mnx / mny)
    return prd


def PRB(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    # not used
    # rtn = None
    if len(x) <= 2:
        rtn = None
    else:
        ratio = x / y
        med = np.median(ratio)
        avmed = x / med
        value = 0.5 * y + 0.5 * avmed
        ind = np.log(value) / np.log(2)
        dep = (ratio - med) / med
        ind2 = sm.add_constant(ind)
        reg = sm.OLS(dep, ind2).fit()
        if reg.pvalues[1] < 0.05:
            rtn = reg.params[1]
        else:
            rtn = 0.0
    return rtn


def PRB_Lower(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    # not used
    # rtn = None
    if len(x) <= 2:
        rtn = None
    else:
        ratio = x / y
        med = np.median(ratio)
        avmed = x / med
        value = 0.5 * y + 0.5 * avmed
        ind = np.log(value) / np.log(2)
        dep = (ratio - med) / med
        ind2 = sm.add_constant(ind)
        reg = sm.OLS(dep, ind2).fit()
        # if reg.pvalues[1]  < .05 :
        #  rtn =  reg.conf_int(alpha=0.05, cols=None)[1,0]
        # else :
        #  rtn = 0
        rtn = reg.conf_int(alpha=0.05, cols=None)[1, 0]
    return rtn


def PRB_Upper(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    # not used
    # rtn = None
    if len(x) <= 2:
        rtn = None
    else:
        ratio = x / y
        med = np.median(ratio)
        avmed = x / med
        value = 0.5 * y + 0.5 * avmed
        ind = np.log(value) / np.log(2)
        dep = (ratio - med) / med
        ind2 = sm.add_constant(ind)
        reg = sm.OLS(dep, ind2).fit()
        # if reg.pvalues[1]  < .05 :
        #  rtn =  reg.conf_int(alpha=0.05, cols=None)[1,1]
        # else :
        #  rtn = 0
        rtn = reg.conf_int(alpha=0.05, cols=None)[1, 1]
    return rtn


def PRB_Conclusion(y: np.ndarray, x: np.ndarray) -> Optional[str]:
    # not used
    # rtn = None
    if len(x) <= 2:
        rtn = None
    else:
        ratio = x / y
        med = np.median(ratio)
        avmed = x / med
        value = 0.5 * y + 0.5 * avmed
        ind = np.log(value) / np.log(2)
        dep = (ratio - med) / med
        ind2 = sm.add_constant(ind)
        reg = sm.OLS(dep, ind2).fit()
        if (
            reg.pvalues[1] > .05 or
            (reg.pvalues[1] <= .05 and np.abs(reg.params[1]) < .05)
        ):
            rtn = 'PASS'
        else:
            rtn = 'FAIL'

    return rtn


def DOR_SUMMARY_Statistics(y: np.ndarray, x: np.ndarray) -> None:
    print(f"Weighted Mean: {weighted_Mean_Sale_Ratio(y, x)}\n")

    if COD(y, x) <= 10:
        print("COD: {COD(y, x)}\n")
    elif COD(y, x) <= 15:
        print(f"COD: {COD(y, x)} <- NOTE THIS IS MODERATELY HIGH\n")
    else:
        print(f"COD: {COD(y, x)} <- NOTE THIS IS ABNORMALY HIGH\n")

    if PRD(y, x) < 0.98 or PRD(y, x) > 1.03:
        print(f"PRD: {PRD(y, x)} <- NOTE THIS IS ABNORMALLY HIGH\n")
    else:
        print(f"PRD: {PRD(y, x)}\n")

    print(
        f"PRB: {PRB(y, x)} <-> "
        f"PRB Lower Bound: {PRB_Lower(y, x)} <-> "
        f"PRB Upper Bound: {PRB_Upper(y, x)} <-> "
        f"PRB RESULT: {PRB_Conclusion(y, x)}"
    )
