
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

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


def VEI (Sale_Amount : np.ndarray, Appr_Value : np.ndarray  , confidence_level = 0.90 , Sales_Ratio_Accuracy = 3 , show_plot = False):
    if isinstance(Appr_Value, pd.Series):
        Sale_Ratios = round(Appr_Value /Sale_Amount ,Sales_Ratio_Accuracy)

        Sale_Ratio_Raw = Appr_Value / Sale_Amount
    else : 
        Sale_Ratios = []
        Sale_Ratio_Raw = []

        for i in range(len(Appr_Value)):
            Sale_Ratio_Raw.append(Appr_Value[i] / Sale_Amount[i])

        Sale_Ratios = [round(value, Sales_Ratio_Accuracy) for value in Sale_Ratio_Raw]

    
    Working_VEI_df = pd.DataFrame({'Appr_Value':Appr_Value,'Sale_Amount':Sale_Amount, 'Raw_Sale_Ratio':Sale_Ratio_Raw  ,'Sale_Ratios':Sale_Ratios})

    Working_VEI_df['Market_Value_Proxy'] = round(.5 * Working_VEI_df['Sale_Amount'] + .5 * (Working_VEI_df['Appr_Value'] / np.median(Working_VEI_df['Raw_Sale_Ratio']) )  ,0)

    Working_VEI_df = Working_VEI_df.sort_values('Market_Value_Proxy',ascending=True)
    
    Working_VEI_df.reset_index(inplace=True,drop=True)

    if Working_VEI_df.shape[0] > 500:
        Group_Count = 10
        Group_Percent = .10
    elif Working_VEI_df.shape[0] > 50 :
        Group_Count = 4
        Group_Percent = .25
    else:
        Group_Count = 2
        Group_Percent = .5

    group_size_list = [] 

    for i in range(1,Group_Count + 1):
        group_size_list.append([math.ceil(Working_VEI_df.shape[0] * Group_Percent * i), i])

    Working_VEI_df['Market_Value_Group'] =  -1 

    prev_value = 0
    for v1, v2 in group_size_list:
        Working_VEI_df.loc[(Working_VEI_df.index + 1 <= v1 ) & (Working_VEI_df.index + 1 > prev_value ),'Market_Value_Group'] = v2
        prev_value = v1

    group_count = []
    group_median = []
    group_Median_Ratio = [] 
    group_Lower_Bound =[]
    group_upper_Bound =[]
    Group_Number = []

    Min_Group_Median = None
    Max_Group_Median = None

    for i in range(1,Working_VEI_df['Market_Value_Group'].max() + 1):
        group_data = Working_VEI_df[Working_VEI_df['Market_Value_Group'] == i].copy()

        group_data.sort_values('Sale_Ratios',inplace=True)   

        group_data.reset_index(inplace=True,drop=True)

        group_count.append(group_data.shape[0])
        group_median.append(np.median(group_data['Market_Value_Proxy']))
        group_Mean = group_data['Market_Value_Proxy'].mean()
        group_n = group_data.shape[0]
        group_std_dev = group_data['Market_Value_Proxy'].std()

        group_Median_Ratio.append( np.median(group_data['Sale_Ratios']) )

        group_median_ratio_index = np.median(list(range(0, group_data.shape[0] )))

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        Temp_Z_Value = z_score * np.sqrt(group_n) / 2
        if group_data.shape[0] % 2 == 0:
            Temp_Z_Value = Temp_Z_Value + .5 
            
        Temp_Z_Value = math.ceil(Temp_Z_Value)

        
        lower_bound = group_data['Sale_Ratios'].iloc[math.ceil(group_median_ratio_index - Temp_Z_Value)] 
        if group_data.shape[0] % 2 == 0:
            upper_bound = group_data['Sale_Ratios'].iloc[math.floor(group_median_ratio_index + Temp_Z_Value)] 
        else: 
            upper_bound = group_data['Sale_Ratios'].iloc[math.ceil(group_median_ratio_index + Temp_Z_Value)] 
        group_Lower_Bound.append(lower_bound)
        group_upper_Bound.append(upper_bound)
        Group_Number.append(i)

        if i == 1:
            Min_Group_Median = np.median(group_data['Sale_Ratios']) 
            Min_Group_LB = lower_bound
            Min_Group_UB = upper_bound

        if i == Working_VEI_df['Market_Value_Group'].max():
            Max_Group_Median = np.median(group_data['Sale_Ratios']) 
            Max_Group_LB = lower_bound
            Max_Group_UB = upper_bound

    Group_VEI_Data = pd.DataFrame({'Group' : Group_Number,'Group_Count':group_count,'Group_Median':group_median,'Group Median_Sale_Ratio' : group_Median_Ratio , 'Group_Lower_Bound':group_Lower_Bound,'Group_Upper_Bound':group_upper_Bound})

    VEI_Score =  100 * ( Max_Group_Median - Min_Group_Median) / np.median( Working_VEI_df['Sale_Ratios'] )

    if Min_Group_Median < Max_Group_Median:
        Lower_Group_Median = Min_Group_Median
        Lower_Group_LB = Min_Group_LB
        Lower_Group_UB = Min_Group_UB
        Upper_Group_Median = Max_Group_Median
        Upper_Group_LB = Max_Group_LB
        Upper_Group_UB = Max_Group_UB
    else:
        Lower_Group_Median = Max_Group_Median
        Lower_Group_LB = Max_Group_LB
        Lower_Group_UB = Max_Group_UB
        Upper_Group_Median = Min_Group_Median
        Upper_Group_LB = Min_Group_LB
        Upper_Group_UB = Min_Group_UB

    VEI_Sig = None

    if Lower_Group_LB <= Upper_Group_UB and Lower_Group_UB >= Upper_Group_LB:
        VEI_Sig = 0 
    else:
        VEI_Sig = 100 * (Upper_Group_LB - Lower_Group_UB) / np.median(Working_VEI_df['Sale_Ratios'] )


    if show_plot == True:
        plt.figure(figsize=(12, 8)) 
        sns.boxplot(data = Working_VEI_df 
                    , y = "Sale_Ratios" 
                    , hue =  'Market_Value_Group'
                    , gap = .2
                    )

        plt.axhline(y= Lower_Group_UB, color='black', linestyle='--')
        plt.axhline(y= Upper_Group_LB, color='black', linestyle='--')
        plt.show()

    return Working_VEI_df, Group_VEI_Data , VEI_Score , VEI_Sig