
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


def _prb_regression(y: np.ndarray, x: np.ndarray):
    """
    Fit the Price-Related Bias (PRB) regression.

    This private helper holds the one shared PRB calculation used by ``PRB``,
    ``PRB_Lower``, ``PRB_Upper`` and ``PRB_Conclusion``, so the regression is
    defined in exactly one place.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResults
        The fitted regression. Position 0 of ``params``, ``pvalues`` and
        ``conf_int()`` is the intercept, and position 1 is the PRB slope.

    Notes
    -----
    Formula, for each sale::

        ratio        = appraised value / sale price                 (x / y)
        median_ratio = median of all ratios
        value_proxy  = 0.5 * sale price + 0.5 * (appraised value / median_ratio)

        independent variable = log2(value_proxy)
        dependent variable   = (ratio - median_ratio) / median_ratio

    An ordinary least squares (OLS) regression is then fitted::

        dependent = intercept + PRB * independent

    The value proxy blends the sale price with the appraised value scaled to
    the overall market level, so neither one alone drives the result. Because
    the independent variable is in log base 2, the PRB slope is the relative
    change in sale ratios each time property value doubles.

    Inputs are converted to plain numpy arrays first. This pairs each sale
    price with its appraised value by position, and makes the regression
    results position-indexed, which works the same with numpy arrays, lists,
    and pandas columns (including filtered DataFrames).
    """
    sale_price = np.asarray(y, dtype=float)
    appraised_value = np.asarray(x, dtype=float)

    ratio = appraised_value / sale_price
    median_ratio = np.median(ratio)
    value_proxy = 0.5 * sale_price + 0.5 * (appraised_value / median_ratio)

    independent = np.log(value_proxy) / np.log(2)  # log base 2
    dependent = (ratio - median_ratio) / median_ratio

    return sm.OLS(dependent, sm.add_constant(independent)).fit()


def PRB(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Price-Related Bias (PRB) coefficient.

    PRB measures vertical equity: whether sale ratios systematically rise
    (progressivity) or fall (regressivity) as property value increases. It is
    the slope of the regression described in ``_prb_regression``: the
    relative change in sale ratios each time property value doubles. For
    example, a PRB of -0.04 means ratios fall about 4% every time value
    doubles, so higher-value properties are relatively under-appraised.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float or None
        The PRB slope when it is statistically significant (p-value < 0.05),
        otherwise ``0.0``. Returns ``None`` when there are 2 or fewer sales,
        which is too few for a regression.

    Notes
    -----
    Use ``PRB_Lower`` and ``PRB_Upper`` for the 95% confidence interval of
    the slope, and ``PRB_Conclusion`` for a PASS / FAIL summary.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> prb = rem.PRB(sales["Sale_Price"], sales["Appraised_Value"])
    """
    if len(x) <= 2:
        return None

    regression = _prb_regression(y, x)
    prb_slope = regression.params[1]
    p_value = regression.pvalues[1]

    if p_value < 0.05:
        return prb_slope
    return 0.0


def PRB_Lower(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Lower bound of the 95% confidence interval for the PRB slope.

    See ``PRB`` for what the Price-Related Bias measures and
    ``_prb_regression`` for the formula.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float or None
        The lower 95% confidence bound of the PRB slope. Unlike ``PRB``, it is
        returned whether or not the slope is statistically significant.
        Returns ``None`` when there are 2 or fewer sales.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.PRB_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    if len(x) <= 2:
        return None

    regression = _prb_regression(y, x)
    # conf_int() returns one row per coefficient: [lower bound, upper bound]
    return regression.conf_int(alpha=0.05)[1, 0]


def PRB_Upper(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Upper bound of the 95% confidence interval for the PRB slope.

    See ``PRB`` for what the Price-Related Bias measures and
    ``_prb_regression`` for the formula.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float or None
        The upper 95% confidence bound of the PRB slope. Unlike ``PRB``, it is
        returned whether or not the slope is statistically significant.
        Returns ``None`` when there are 2 or fewer sales.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.PRB_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    if len(x) <= 2:
        return None

    regression = _prb_regression(y, x)
    # conf_int() returns one row per coefficient: [lower bound, upper bound]
    return regression.conf_int(alpha=0.05)[1, 1]


def PRB_Conclusion(y: np.ndarray, x: np.ndarray) -> Optional[str]:
    """
    PASS / FAIL summary of the Price-Related Bias (PRB) test.

    See ``PRB`` for what the Price-Related Bias measures and
    ``_prb_regression`` for the formula.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    str or None
        ``'PASS'`` or ``'FAIL'``. Returns ``None`` when there are 2 or fewer
        sales.

    Notes
    -----
    Decision rule:

    * ``'PASS'`` if the PRB slope is not statistically significant
      (p-value > 0.05), or if it is significant but small
      (between -0.05 and 0.05).
    * ``'FAIL'`` otherwise. That is, a statistically significant slope of
      0.05 or more in either direction.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.PRB_Conclusion(sales["Sale_Price"], sales["Appraised_Value"])
    """
    if len(x) <= 2:
        return None

    regression = _prb_regression(y, x)
    prb_slope = regression.params[1]
    p_value = regression.pvalues[1]

    # Note: if the p-value is undefined (NaN), for example when every sale
    # ratio is identical, neither condition below is true and the result is
    # 'FAIL'. The condition is written out in full to keep that behavior.
    if (
        p_value > 0.05 or
        (p_value <= 0.05 and np.abs(prb_slope) < 0.05)
    ):
        return 'PASS'
    return 'FAIL'


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