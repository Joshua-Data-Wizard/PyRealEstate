"""
Ratio-study statistics for mass appraisal: weighted mean sale ratio, COD,
PRD, PRB, and VEI.

Argument convention used by every function in this module::

    y  (VEI: Sale_Amount) : sale prices
    x  (VEI: Appr_Value)  : appraised / model-predicted values
    sale ratio            = appraised value / sale price

Swapping the two arguments silently inverts every ratio, so always pass the
sale prices first.

Data checks
-----------
Every function first checks the sales it is given, pairing each sale price
with the appraised value in the same position:

* A sale with a missing (NaN) or infinite sale price or appraised value, or a
  sale price of 0 or less, cannot form a sale ratio. It is left out, and a
  ``DataQualityWarning`` reports how many sales of each kind were left out
  and in which rows.
* A sale with an appraised value of 0 or less is kept, and a
  ``DataQualityWarning`` reports how many of each kind there are and in which
  rows. The exception is ``VEI``: its confidence interval uses the logarithm
  of each sale ratio, so it leaves these sales out, and reports them.
* Inputs of different lengths, empty inputs, inputs that are not numbers, or
  inputs with no usable sales raise a ``ValueError``.

Rows are identified by the pandas index when a pandas Series is given, and
otherwise by position (starting at 0).
"""

import math
import numbers
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels, scipy, and matplotlib are imported inside the functions that
# use them. Each takes up to ~1.5 s to import, and users who only need the
# weighted mean, COD, or PRD should not have to wait for them.

from ._data_checks import (
    DataQualityWarning,
    _problem_lines,
    _reissue_once,
    _row_labels,
    _warn_data_quality,
)


def _sales_pairs(
    y: Any,
    x: Any,
    return_labels: bool = False,
    report_unusual_appraised: bool = True,
) -> Tuple[np.ndarray, ...]:
    """
    Check sale prices and appraised values, and return the usable sales.

    Each sale price is paired with the appraised value in the same position,
    whatever the input type (numpy array, list, or pandas Series with any
    index). See "Data checks" in the module documentation for what is left
    out, what is reported, and what raises an error.

    Parameters
    ----------
    y : array-like
        Sale prices.
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    return_labels : bool, default False
        Also return the row label of each usable sale (the pandas index, or
        the position), so the caller can name rows or label its results.
    report_unusual_appraised : bool, default True
        Warn about appraised values of 0 or less (which are kept). ``VEI``
        turns this off because it leaves those sales out and reports them
        itself.

    Returns
    -------
    sale_price, appraised_value : numpy.ndarray
        The usable sales, paired by position. pandas decides the number type,
        so for example a list of whole numbers becomes int64 on every platform.
    row_labels : numpy.ndarray
        Only when ``return_labels`` is True: the row label of each usable sale.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, are not numbers, or contain
        no usable sales.

    Warns
    -----
    DataQualityWarning
        If sales are left out, or if any appraised values are 0 or negative.
    """
    sale_price = pd.Series(y).to_numpy()
    appraised_value = pd.Series(x).to_numpy()

    if len(sale_price) != len(appraised_value):
        raise ValueError(
            f"There are {len(sale_price)} sale prices (y) but {len(appraised_value)} "
            "appraised values (x); each sale needs exactly one appraised value."
        )
    if len(sale_price) == 0:
        raise ValueError(
            "No sales were provided: the sale prices (y) and appraised values (x) are empty."
        )
    for values, name in ((sale_price, "Sale prices (y)"), (appraised_value, "Appraised values (x)")):
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{name} must be numbers.")

    # Rows are named by the pandas index when one was given, otherwise by position.
    row_labels = _row_labels((y, x), len(sale_price))

    sale_float = sale_price.astype(float)
    appraised_float = appraised_value.astype(float)

    # Sales that cannot form a sale ratio are left out.
    cannot_use = [
        ("missing sale price", np.isnan(sale_float)),
        ("infinite sale price", np.isinf(sale_float)),
        ("sale price of 0", sale_float == 0),
        ("negative sale price", np.isfinite(sale_float) & (sale_float < 0)),
        ("missing appraised value", np.isnan(appraised_float)),
        ("infinite appraised value", np.isinf(appraised_float)),
    ]
    left_out = np.logical_or.reduce([mask for _, mask in cannot_use])
    if left_out.all():
        raise ValueError(
            f"None of the {len(sale_price)} sales can form a sale ratio:\n"
            + _problem_lines(cannot_use, row_labels)
        )
    if left_out.any():
        _warn_data_quality(
            f"pyRealEstate left out {int(left_out.sum())} of {len(sale_price)} sales "
            "that cannot form a sale ratio:\n"
            + _problem_lines(cannot_use, row_labels)
            + f"\nThe statistics use the remaining {int((~left_out).sum())} sales."
        )

    keep = ~left_out
    sale_price, appraised_value = sale_price[keep], appraised_value[keep]
    appraised_float, row_labels = appraised_float[keep], row_labels[keep]

    # Unusual appraised values are kept, but reported.
    unusual = [
        ("appraised value of 0", appraised_float == 0),
        ("negative appraised value", appraised_float < 0),
    ]
    if report_unusual_appraised and any(mask.any() for _, mask in unusual):
        _warn_data_quality(
            "pyRealEstate found unusual appraised values. These sales are included "
            "in the statistics; please check that they are correct:\n"
            + _problem_lines(unusual, row_labels)
        )

    if return_labels:
        return sale_price, appraised_value, row_labels
    return sale_price, appraised_value


def weighted_Mean_Sale_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Weighted mean sale ratio.

    The total of the appraised values divided by the total of the sale
    prices. Because it works with totals, higher-priced sales carry more
    weight than in the simple average of the sale ratios.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The weighted mean sale ratio.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Notes
    -----
    Formula::

        weighted mean = sum(appraised values) / sum(sale prices)
                      = mean(x) / mean(y)

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.weighted_Mean_Sale_Ratio(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    return np.mean(x) / np.mean(y)


def COD(y: np.ndarray, x: np.ndarray) -> float:
    """
    Coefficient of Dispersion (COD).

    The average absolute difference between each sale ratio and the median
    ratio, as a percentage of the median ratio. It measures uniformity: the
    lower the COD, the more consistently properties are appraised relative to
    their sale prices.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The COD, as a percentage (e.g. 7.0 means 7%).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Notes
    -----
    Formula::

        ratio = appraised value / sale price                    (x / y)
        COD   = 100 * mean(|ratio - median ratio|) / median ratio

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.COD(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    ratio = x / y
    med = np.median(ratio)
    dev = np.sum(np.abs(ratio - med))
    avgdev = dev / len(ratio)
    cod = 100 * (avgdev / med)
    return cod


def PRD(y: np.ndarray, x: np.ndarray) -> float:
    """
    Price-Related Differential (PRD).

    The mean sale ratio divided by the weighted mean sale ratio. It measures
    vertical equity. A PRD above 1 suggests higher-value properties are
    appraised at relatively lower ratios than lower-value properties
    (regressivity); a PRD below 1 suggests the opposite (progressivity).

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The PRD.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Notes
    -----
    Formula::

        ratio = appraised value / sale price                    (x / y)
        PRD   = mean(ratio) / (mean(x) / mean(y))
              = mean ratio / weighted mean ratio

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.PRD(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
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
    defined in exactly one place. The callers check the data first with
    ``_sales_pairs``.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResults or None
        The fitted regression. Position 0 of ``params``, ``pvalues`` and
        ``conf_int()`` is the intercept, and position 1 is the PRB slope.
        ``None`` when 2 or fewer sales can be used, which is too few for a
        regression.

    Warns
    -----
    DataQualityWarning
        If sales are left out because their value proxy is 0 or negative.

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

    The logarithm needs a value proxy above 0. Only a negative appraised value
    can make it 0 or negative; such sales are left out of the regression (the
    median ratio still uses every sale), and a ``DataQualityWarning`` says how
    many.

    Inputs are converted to plain numpy arrays first. This pairs each sale
    price with its appraised value by position, and makes the regression
    results position-indexed, which works the same with numpy arrays, lists,
    and pandas columns (including filtered DataFrames).
    """
    import statsmodels.api as sm  # imported here: slow to import (see top of module)

    sale_price = np.asarray(y, dtype=float)
    appraised_value = np.asarray(x, dtype=float)

    ratio = appraised_value / sale_price
    median_ratio = np.median(ratio)
    value_proxy = 0.5 * sale_price + 0.5 * (appraised_value / median_ratio)

    usable = value_proxy > 0
    if not usable.all():
        _warn_data_quality(
            f"PRB left out {int((~usable).sum())} of {len(usable)} sales whose value "
            "proxy (0.5 x sale price + 0.5 x appraised value / median ratio) is 0 or "
            "negative, because PRB uses its logarithm. This only happens with "
            "negative appraised values."
        )
    if usable.sum() <= 2:
        return None

    independent = np.log(value_proxy[usable]) / np.log(2)  # log base 2
    dependent = (ratio[usable] - median_ratio) / median_ratio

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
        otherwise ``0.0``. Returns ``None`` when 2 or fewer sales can be used,
        which is too few for a regression.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Notes
    -----
    Use ``PRB_Lower`` and ``PRB_Upper`` for the 95% confidence interval of
    the slope, and ``PRB_Conclusion`` for a PASS / FAIL summary.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> prb = rem.PRB(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None

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
        Returns ``None`` when 2 or fewer sales can be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.PRB_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None

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
        Returns ``None`` when 2 or fewer sales can be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.PRB_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None

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
        ``'PASS'`` or ``'FAIL'``. Returns ``None`` when 2 or fewer sales can
        be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

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
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None

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


def DOR_SUMMARY_Statistics(
    y: np.ndarray,
    x: np.ndarray,
    return_results: bool = False
) -> Optional[dict]:
    """
    Print a summary of the main ratio-study statistics.

    Calculates the weighted mean sale ratio, COD, PRD, and PRB (with its 95%
    confidence interval and PASS / FAIL conclusion), prints them, and flags
    COD and PRD values that fall outside common target ranges.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    return_results : bool, default False
        If True, also return the statistics as a dictionary, so they can be
        used in further calculations or saved.

    Returns
    -------
    dict or None
        When ``return_results`` is True, a dictionary with the keys
        ``'Weighted_Mean'``, ``'COD'``, ``'PRD'``, ``'PRB'``, ``'PRB_Lower'``,
        ``'PRB_Upper'`` and ``'PRB_Conclusion'``. Otherwise ``None``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation). Each statistic checks the data,
        but each problem is reported once.

    Notes
    -----
    Each statistic is calculated by the function of the same name in this
    module (``weighted_Mean_Sale_Ratio``, ``COD``, ``PRD``, ``PRB``,
    ``PRB_Lower``, ``PRB_Upper``, ``PRB_Conclusion``). See their
    documentation for the formulas.

    The printed notes use these ranges:

    * COD above 10 is noted as moderately high, and above 15 as abnormally
      high.
    * PRD below 0.98 is noted as abnormally low, and above 1.03 as abnormally
      high.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.DOR_SUMMARY_Statistics(sales["Sale_Price"], sales["Appraised_Value"])
    >>> results = rem.DOR_SUMMARY_Statistics(
    ...     sales["Sale_Price"], sales["Appraised_Value"], return_results=True
    ... )
    >>> results["COD"]
    """
    # Calculate each statistic once, then print. Each statistic checks the
    # data itself, so collect their data-quality warnings and report each
    # distinct problem once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataQualityWarning)
        weighted_mean = weighted_Mean_Sale_Ratio(y, x)
        cod = COD(y, x)
        prd = PRD(y, x)
        prb = PRB(y, x)
        prb_lower = PRB_Lower(y, x)
        prb_upper = PRB_Upper(y, x)
        prb_conclusion = PRB_Conclusion(y, x)
    _reissue_once(caught)

    print(f"Weighted Mean: {weighted_mean}\n")

    if cod <= 10:
        print(f"COD: {cod}\n")
    elif cod <= 15:
        print(f"COD: {cod} <- NOTE THIS IS MODERATELY HIGH\n")
    else:
        print(f"COD: {cod} <- NOTE THIS IS ABNORMALLY HIGH\n")

    if prd < 0.98:
        print(f"PRD: {prd} <- NOTE THIS IS ABNORMALLY LOW\n")
    elif prd > 1.03:
        print(f"PRD: {prd} <- NOTE THIS IS ABNORMALLY HIGH\n")
    else:
        print(f"PRD: {prd}\n")

    print(
        f"PRB: {prb} <-> "
        f"PRB Lower Bound: {prb_lower} <-> "
        f"PRB Upper Bound: {prb_upper} <-> "
        f"PRB RESULT: {prb_conclusion}"
    )

    if return_results:
        return {
            'Weighted_Mean': weighted_mean,
            'COD': cod,
            'PRD': prd,
            'PRB': prb,
            'PRB_Lower': prb_lower,
            'PRB_Upper': prb_upper,
            'PRB_Conclusion': prb_conclusion,
        }
    return None


# ---------------------------------------------------------------------------
# Vertical Equity Indicator (VEI)
#
# Source: IAAO Standard on Ratio Studies (2026), Appendix E. Section, step and
# table numbers in this part of the module (E.3, Step 6, Table E.7.1, ...)
# refer to that appendix.
# ---------------------------------------------------------------------------

_VEI_MINIMUM_SALES = 20  # Table E.3.1: with fewer than 20 sales the VEI is not calculated
_VEI_LIMIT = 10.0        # Table E.7.1: a VEI within +/-10% is acceptable

# The conclusions returned by VEI and VEI_Conclusion
_VEI_REGRESSIVE = "Regressive"
_VEI_PROGRESSIVE = "Progressive"
_VEI_NOT_SIGNIFICANT = "No significant progressivity or regressivity"
_VEI_NOT_ENOUGH_SALES = f"Not enough sales (fewer than {_VEI_MINIMUM_SALES})"

_VEI_GROUP_COLUMNS = [
    "Group", "Sales", "Median_Proxy", "Median_Ratio", "COD",
    "SD_Ln_Ratio", "SE_Ln_Median", "SE_Median",
]


@dataclass(frozen=True, eq=False, repr=False)
class VEIResult:
    """
    The result of ``VEI``: the Vertical Equity Indicator, its confidence
    interval, the Standard's verdict, and the tables behind them.

    ``print(result)`` shows a plain-English report, and ``result.plot()``
    draws a chart. The values are read-only.

    Attributes
    ----------
    vei : float
        The VEI point estimate, in percent (48.53 means 48.53%). Positive
        means the highest-value group has the higher sale ratios
        (progressive); negative means it has the lower ratios (regressive).
        NaN when there are fewer than 20 sales.
    ci_lower, ci_upper : float
        The confidence interval of the VEI, in percent. Always reported, even
        when the VEI is within +/-10% and the Standard does not need it.
    confidence_level : float
        The confidence level of the interval (e.g. 0.90).
    scenario : int or None
        Which row of the Standard's Table E.7.1 applies, 1 to 4 (see the
        Notes of ``VEI``). None when there are fewer than 20 sales.
    conclusion : str
        ``'Regressive'``, ``'Progressive'``, ``'No significant progressivity
        or regressivity'``, or ``'Not enough sales (fewer than 20)'``.
    overall_median : float
        The median sale ratio of all the sales used.
    ranking_method : str
        ``'R7'`` or ``'R6'``: how sales were assigned to groups.
    groups : pandas.DataFrame
        One row per value group, lowest value first:

        * ``Group``: the group number, 1 for the lowest values;
        * ``Sales``: the number of sales in the group;
        * ``Median_Proxy``: the group's median market value proxy;
        * ``Median_Ratio``: the group's median sale ratio;
        * ``COD``: the group's coefficient of dispersion, in percent;
        * ``SD_Ln_Ratio``: the sample standard deviation of ln(ratio);
        * ``SE_Ln_Median``: the standard error of the median, log scale;
        * ``SE_Median``: the standard error of the median, ratio scale.

        Empty when there are fewer than 20 sales.
    sales : pandas.DataFrame
        One row per sale used, sorted by the proxy, with the columns
        ``Sale_Amount``, ``Appr_Value``, ``Sale_Ratio``,
        ``Market_Value_Proxy`` and ``Group``. Rows keep the labels of the
        input (the pandas index, or the position for numpy arrays and lists),
        so ``your_sales.join(result.sales["Group"])`` adds each sale's group
        to your own table. ``Group`` is empty (NaN) when there are fewer than
        20 sales.
    notes : tuple of str
        Points the Standard says to review (Appendix E.4): an inner group
        with the lowest or highest median ratio. Usually empty.
    """

    vei: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    scenario: Optional[int]
    conclusion: str
    overall_median: float
    ranking_method: str
    groups: pd.DataFrame
    sales: pd.DataFrame
    notes: Tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (
            f"VEIResult(vei={self.vei:.2f}, ci_lower={self.ci_lower:.2f}, "
            f"ci_upper={self.ci_upper:.2f}, scenario={self.scenario}, "
            f"conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        title = "Vertical Equity Indicator (VEI), IAAO Standard on Ratio Studies (2026), Appendix E"
        if self.scenario is None:
            return "\n".join([
                title,
                f"{len(self.sales)} usable sales; the VEI needs at least {_VEI_MINIMUM_SALES}.",
                f"Conclusion: {self.conclusion}",
            ])

        group_table = self.groups[["Group", "Sales", "Median_Proxy", "Median_Ratio", "COD"]].to_string(
            index=False,
            formatters={
                "Median_Proxy": "{:,.0f}".format,
                "Median_Ratio": "{:.4f}".format,
                "COD": "{:.2f}".format,
            },
        )
        lines = [
            title,
            f"{len(self.sales)} sales in {len(self.groups)} groups ({self.ranking_method} ranking), "
            f"overall median ratio {self.overall_median:.4f}",
            "",
            f"VEI: {self.vei:.2f}%",
            f"{_percent(self.confidence_level)} confidence interval: "
            f"{self.ci_lower:.2f}% to {self.ci_upper:.2f}%",
            textwrap.fill(_vei_scenario_text(self.scenario, self.vei), width=88),
            f"Conclusion: {self.conclusion}",
            "",
            group_table,
        ]
        lines += [textwrap.fill(f"Note: {note}", width=88) for note in self.notes]
        return "\n".join(lines)

    def plot(self, ax: Optional["matplotlib.axes.Axes"] = None, show: bool = True) -> None:
        """
        Box plot of the sale ratios in each value group.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Draw on these axes (e.g. one panel of your own figure) instead of
            a new figure.
        show : bool, default True
            Display the plot. Use False to keep customizing or to save it, for
            example with ``plt.savefig("vei.png", dpi=300)``.
        """
        import matplotlib.pyplot as plt  # imported here: slow to import (see top of module)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if self.scenario is None:
            ax.set_title(f"VEI: {self.conclusion}")
        else:
            group_numbers = self.groups["Group"].tolist()
            ratios_by_group = [
                self.sales.loc[self.sales["Group"] == group, "Sale_Ratio"].to_numpy()
                for group in group_numbers
            ]
            positions = list(range(1, len(group_numbers) + 1))
            ax.boxplot(ratios_by_group, positions=positions)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"Group {group}" for group in group_numbers])
            ax.axhline(
                self.overall_median, color="gray", linestyle="--", linewidth=1,
                label=f"Overall median ratio ({self.overall_median:.4f})",
            )
            ax.legend(loc="best")
            ax.set_title(
                f"VEI {self.vei:.2f}% ({_percent(self.confidence_level)} CI "
                f"{self.ci_lower:.2f}% to {self.ci_upper:.2f}%): {self.conclusion}"
            )

        ax.set_xlabel("Value group (by market value proxy, lowest to highest)")
        ax.set_ylabel("Sale ratio (appraised value / sale price)")
        if show:
            plt.show()


def _percent(fraction: float) -> str:
    """0.9 -> '90%', 0.975 -> '97.5%'."""
    return f"{fraction * 100:g}%"


def _vei_scenario_text(scenario: int, vei: float) -> str:
    """One sentence explaining which row of Table E.7.1 applies."""
    tendency = "progressive" if vei > 0 else "regressive"
    return {
        1: "Scenario 1 (Table E.7.1): the VEI is within +/-10%, so vertical equity is acceptable.",
        2: (
            f"Scenario 2 (Table E.7.1): the VEI is outside +/-10%, a {tendency} tendency, but its "
            "confidence interval overlaps +/-10%, so the difference is not statistically "
            "significant. Vertical equity is acceptable."
        ),
        3: (
            "Scenario 3 (Table E.7.1): the VEI and its whole confidence interval are below -10%. "
            "The level of regressivity does not comply with the Standard."
        ),
        4: (
            "Scenario 4 (Table E.7.1): the VEI and its whole confidence interval are above +10%. "
            "The level of progressivity does not comply with the Standard."
        ),
    }[scenario]


def _check_vei_options(confidence_level: float, ranking_method: str) -> str:
    """Check the VEI's options and return the ranking method as 'R6' or 'R7'."""
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, numbers.Real)
        or not 0 < confidence_level < 1  # also catches NaN
    ):
        raise ValueError(
            "confidence_level must be a number between 0 and 1 (for example 0.90 for 90%), "
            f"not {confidence_level!r}."
        )
    method = ranking_method.strip().upper() if isinstance(ranking_method, str) else None
    if method not in ("R6", "R7"):
        raise ValueError(
            "ranking_method must be 'R6' or 'R7' (Standard on Ratio Studies, Appendix E.5), "
            f"not {ranking_method!r}."
        )
    return method


def _vei_leave_out_ratios_of_zero_or_less(
    sale_price: np.ndarray, appraised_value: np.ndarray, row_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave out sales with an appraised value of 0 or less, and report them.

    Their sale ratio is 0 or negative, and the VEI's confidence interval
    (Step 6) uses the natural logarithm of each ratio, which only exists for
    ratios above 0. (The other statistics keep these sales, with a warning.)
    """
    appraised_float = appraised_value.astype(float)
    problems = [
        ("appraised value of 0", appraised_float == 0),
        ("negative appraised value", appraised_float < 0),
    ]
    left_out = appraised_float <= 0
    if left_out.any():
        n_left_out, n_sales = int(left_out.sum()), len(left_out)
        _warn_data_quality(
            f"The VEI left out {n_left_out} of {n_sales} sales whose appraised value is 0 or "
            "negative. Its confidence interval uses the logarithm of each sale ratio, which only "
            "exists for ratios above 0:\n"
            + _problem_lines(problems, row_labels)
            + f"\nThe VEI uses the remaining {n_sales - n_left_out} sales."
        )
    keep = ~left_out
    return sale_price[keep], appraised_value[keep], row_labels[keep]


def _vei_number_of_groups(n_sales: int) -> int:
    """Table E.3.1: 20-50 sales -> 2 groups (halves), 51-500 -> 4 (quartiles), 501+ -> 10 (deciles)."""
    if n_sales <= 50:
        return 2
    if n_sales <= 500:
        return 4
    return 10


def _vei_group_numbers(n_sales: int, ranking_method: str) -> np.ndarray:
    """
    The group number (1, 2, ...) of each sale, in order of the proxy.

    Section E.5: for group i of k, the upper percentile boundary is
    P = 100 * i / k, and the group's last sale is at position h, rounded
    down, where::

        R7:  h = P * (n - 1) / 100 + 1
        R6:  h = P * (n + 1) / 100        (the last group ends at n)

    With P = 100 * i / k these become whole-number divisions, which ``//``
    computes exactly (no floating-point rounding just below a whole number)::

        R7:  (i * (n - 1)) // k + 1
        R6:  (i * (n + 1)) // k
    """
    n_groups = _vei_number_of_groups(n_sales)
    group = np.arange(1, n_groups + 1, dtype=np.int64)  # int64 on every platform (F-122)
    if ranking_method == "R7":
        group_ends = group * (n_sales - 1) // n_groups + 1
    else:
        group_ends = group * (n_sales + 1) // n_groups
    group_ends[-1] = n_sales  # R7 already gives n; R6 uses n instead of n + 1
    group_sizes = np.diff(group_ends, prepend=0)
    return np.repeat(group, group_sizes)


def _vei_group_table(sales: pd.DataFrame) -> pd.DataFrame:
    """
    The statistics of each value group: Table E.6.1 (sales, median proxy,
    median ratio, COD) and the standard errors of Step 6b and 6c.
    """
    sales = sales.reset_index(drop=True)  # work by position: input row labels may repeat
    group = sales["Group"]
    ratio = sales["Sale_Ratio"]
    n_sales = ratio.groupby(group).size()
    median_ratio = ratio.groupby(group).median()

    # COD = 100 * average absolute difference from the group median / group median
    absolute_difference = (ratio - group.map(median_ratio)).abs()
    cod = 100 * absolute_difference.groupby(group).mean() / median_ratio

    # Step 6b: standard error of the median on the log scale, using the sample
    # standard deviation of ln(ratio) (pandas' std divides by n - 1).
    sd_ln_ratio = np.log(ratio).groupby(group).std()
    se_ln_median = math.sqrt(math.pi / 2) * sd_ln_ratio / np.sqrt(n_sales)
    # Step 6c: back to the ratio scale
    se_median = median_ratio * se_ln_median

    return pd.DataFrame({
        "Group": median_ratio.index.to_numpy(),
        "Sales": n_sales.to_numpy(),
        "Median_Proxy": sales["Market_Value_Proxy"].groupby(group).median().to_numpy(),
        "Median_Ratio": median_ratio.to_numpy(),
        "COD": cod.to_numpy(),
        "SD_Ln_Ratio": sd_ln_ratio.to_numpy(),
        "SE_Ln_Median": se_ln_median.to_numpy(),
        "SE_Median": se_median.to_numpy(),
    })


def _vei_verdict(vei: float, ci_lower: float, ci_upper: float) -> Tuple[int, str]:
    """
    The scenario (1-4) and conclusion from Table E.7.1.

    Values are compared after rounding to 10 decimal places. A VEI that is
    mathematically exactly 10% (common with round-number prices, e.g. group
    medians of 0.95 and 1.05 and an overall median of 1.00) comes out of
    floating-point arithmetic as 10.000000000000009, and must still count as
    "within +/-10%".
    """
    vei, ci_lower, ci_upper = (round(value, 10) for value in (vei, ci_lower, ci_upper))
    if -_VEI_LIMIT <= vei <= _VEI_LIMIT:
        return 1, _VEI_NOT_SIGNIFICANT  # phase 1: acceptable, no further test needed
    if ci_upper < -_VEI_LIMIT:
        return 3, _VEI_REGRESSIVE
    if ci_lower > _VEI_LIMIT:
        return 4, _VEI_PROGRESSIVE
    return 2, _VEI_NOT_SIGNIFICANT


def _vei_notes(groups: pd.DataFrame) -> Tuple[str, ...]:
    """
    Appendix E.4: "If another quantile group has the highest or lowest median
    ratio, it should be noted and reviewed to identify any correctable issues."
    """
    if len(groups) <= 2:
        return ()  # no inner groups
    first_and_last = groups["Median_Ratio"].iloc[[0, -1]]
    inner = groups.iloc[1:-1]
    lowest = inner.loc[inner["Median_Ratio"].idxmin()]
    highest = inner.loc[inner["Median_Ratio"].idxmax()]

    notes = []
    for row, word, is_extreme in (
        (lowest, "lowest", lowest["Median_Ratio"] < first_and_last.min()),
        (highest, "highest", highest["Median_Ratio"] > first_and_last.max()),
    ):
        if is_extreme:
            notes.append(
                f"Group {int(row['Group'])} has the {word} median ratio "
                f"({row['Median_Ratio']:.4f}), but the VEI compares only the first and last "
                "groups. The Standard (Appendix E.4) recommends noting this and reviewing the "
                "group for correctable issues."
            )
    return tuple(notes)


def _vei_calculation(
    sale_price: np.ndarray,
    appraised_value: np.ndarray,
    row_labels: np.ndarray,
    confidence_level: float,
    ranking_method: str,
) -> VEIResult:
    """The VEI calculation (Appendix E.3) on sales that have passed the data checks."""
    n_sales = len(sale_price)

    # Step 1: the sale ratio of each sale (not rounded).
    ratio = appraised_value / sale_price
    overall_median = float(np.median(ratio)) if n_sales else math.nan

    # Step 2: the market value proxy, written exactly as in the Standard.
    proxy = 0.5 * sale_price + 0.5 * (appraised_value / overall_median)

    # Step 3: sort by the proxy. The sort is stable, so sales with the same
    # proxy keep their input order (the Standard does not specify an order).
    order = np.argsort(proxy, kind="stable")
    sales = pd.DataFrame(
        {
            "Sale_Amount": sale_price[order],
            "Appr_Value": appraised_value[order],
            "Sale_Ratio": ratio[order],
            "Market_Value_Proxy": proxy[order],
        },
        index=row_labels[order],
    )

    if n_sales < _VEI_MINIMUM_SALES:
        _warn_data_quality(
            f"The VEI needs at least {_VEI_MINIMUM_SALES} sales (IAAO Standard on Ratio Studies, "
            f"Table E.3.1), but only {n_sales} usable sales were given. The VEI, its confidence "
            "interval and the group results are left empty (NaN)."
        )
        sales["Group"] = np.nan
        return VEIResult(
            vei=math.nan, ci_lower=math.nan, ci_upper=math.nan,
            confidence_level=confidence_level, scenario=None,
            conclusion=_VEI_NOT_ENOUGH_SALES, overall_median=overall_median,
            ranking_method=ranking_method,
            groups=pd.DataFrame(columns=_VEI_GROUP_COLUMNS), sales=sales,
        )

    # Step 3 (continued) and Step 4: the groups and their statistics.
    sales["Group"] = _vei_group_numbers(n_sales, ranking_method)
    groups = _vei_group_table(sales)
    first, last = groups.iloc[0], groups.iloc[-1]

    # Step 5: the VEI point estimate.
    difference = last["Median_Ratio"] - first["Median_Ratio"]
    vei = 100 * difference / overall_median

    # Step 6: the confidence interval of the VEI.
    from scipy import stats  # imported here: slow to import (see top of module)

    se_difference = math.sqrt(first["SE_Median"] ** 2 + last["SE_Median"] ** 2)  # Step 6d
    z = stats.norm.ppf((1 + confidence_level) / 2)  # two-sided, e.g. 1.645 for 90%
    ci_lower = 100 * (difference - z * se_difference) / overall_median          # Step 6e
    ci_upper = 100 * (difference + z * se_difference) / overall_median

    scenario, conclusion = _vei_verdict(vei, ci_lower, ci_upper)
    return VEIResult(
        vei=float(vei), ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        confidence_level=confidence_level, scenario=scenario, conclusion=conclusion,
        overall_median=overall_median, ranking_method=ranking_method,
        groups=groups, sales=sales, notes=_vei_notes(groups),
    )


def VEI(
    Sale_Amount: np.ndarray,
    Appr_Value: np.ndarray,
    confidence_level: float = 0.90,
    ranking_method: str = "R7",
    show_plot: bool = False,
) -> VEIResult:
    """
    Vertical Equity Indicator (VEI), following the IAAO Standard on Ratio
    Studies (2026), Appendix E.

    The VEI checks vertical equity: whether lower- and higher-value properties
    are appraised at the same level. It sorts the sales by a market value
    proxy, splits them into value groups, and compares the median sale ratio
    of the highest-value group with that of the lowest-value group. Its
    confidence interval shows whether a difference is statistically
    significant, and the conclusion follows the Standard's Table E.7.1.

    .. versionchanged:: 0.3.0
       Rewritten to follow the 2026 Standard, and the results differ from
       earlier versions. ``VEI`` now returns a ``VEIResult`` instead of the
       4-tuple ``(Working_VEI_df, Group_VEI_Data, VEI_Score, VEI_Sig)``.
       ``VEI_Sig`` is replaced by the confidence interval of the VEI itself.
       Ratios are no longer rounded, so ``Sales_Ratio_Accuracy`` was removed.
       Sales are grouped with the Standard's R7 (or R6) ranking method.
       Earlier versions followed a draft of the method that the IAAO revised
       before adopting it. That draft method remains in pyRealEstate 0.2.1
       only for reproducing past results; do not use it for new ratio studies
       (0.2.1 also lacks the other fixes made in 0.3.0).

    Parameters
    ----------
    Sale_Amount : array-like
        Sale prices (numpy array, list, or pandas Series). Use time-adjusted
        sale prices if the sales have been adjusted for time.
    Appr_Value : array-like
        Appraised / model-predicted values, in the same order as
        ``Sale_Amount``.
    confidence_level : float, default 0.90
        Confidence level of the VEI's confidence interval, between 0 and 1.
        The Standard's examples use 90%.
    ranking_method : {"R7", "R6"}, default "R7"
        How sales are assigned to groups when they don't divide evenly
        (Section E.5). R7 is used in the Standard's worked example and is the
        default in Python (numpy), R and Excel; R6 is the default in SPSS and
        NCSS. Upper or lower case is fine.
    show_plot : bool, default False
        If True, show a box plot of the sale ratios in each value group (see
        ``VEIResult.plot``).

    Returns
    -------
    VEIResult
        The VEI (``vei``), its confidence interval (``ci_lower``,
        ``ci_upper``), the scenario of Table E.7.1 (``scenario``), the
        conclusion (``conclusion``), the group table (``groups``), each sale's
        ratio, proxy and group (``sales``), and points to review (``notes``).
        See ``VEIResult``. ``print(result)`` shows a plain-English report.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, are not numbers, or have no
        usable sales; if ``confidence_level`` is not between 0 and 1; or if
        ``ranking_method`` is not "R6" or "R7".

    Warns
    -----
    DataQualityWarning
        If sales are left out (see "Data checks" in the module documentation),
        including sales with an appraised value of 0 or less, which the VEI
        cannot use; or if fewer than 20 usable sales remain.

    See Also
    --------
    VEI_Lower, VEI_Upper, VEI_Conclusion : one part of the result at a time.

    Notes
    -----
    The calculation (Appendix E.3)::

        1. ratio  = appraised value / sale price                  (not rounded)
        2. proxy  = 0.5 * sale price + 0.5 * appraised value / overall median ratio
        3. Sort the sales by proxy, lowest first, and split them into groups:
               20 to 50 sales   -> 2 groups (halves)
               51 to 500 sales  -> 4 groups (quartiles)
               501 or more      -> 10 groups (deciles)
           With fewer than 20 sales the VEI is not calculated (Table E.3.1).
        4. The median ratio of each group.
        5. VEI = 100 * (median of the last group - median of the first group)
                     / overall median ratio
        6. The confidence interval:
               SE_ln(group)  = sqrt(pi / 2) * SD(ln ratios in the group) / sqrt(n)
               SE(group)     = median ratio of the group * SE_ln(group)
               SE_difference = sqrt(SE(first group)^2 + SE(last group)^2)
               CI = 100 * (difference -/+ z * SE_difference) / overall median ratio
           SD is the sample standard deviation, and z is the two-sided normal
           critical value for ``confidence_level`` (1.645 for 90%).

    The conclusion (Table E.7.1):

    * Scenario 1, the VEI is within +/-10%: acceptable, "No significant
      progressivity or regressivity".
    * Scenario 2, the VEI is outside +/-10% but its confidence interval
      overlaps +/-10%: acceptable, "No significant progressivity or
      regressivity". There is a tendency, but it is not statistically
      significant; the printed report names it.
    * Scenario 3, the whole interval is below -10%: "Regressive".
    * Scenario 4, the whole interval is above +10%: "Progressive".

    A positive VEI means the highest-value group has the higher sale ratios:
    higher-value properties are appraised at a larger share of their market
    value than lower-value properties, which is progressivity. A negative VEI
    means regressivity (Section 8.2 and Appendix E.3, Step 5).

    The Standard calculates the confidence interval only when the VEI is
    outside +/-10%. This function always reports it, because it helps judge
    the result. The conclusion is the same either way: the interval is
    centered on the VEI, so a VEI within +/-10% always has an interval that
    overlaps +/-10%.

    Groups (Section E.5): with n sales in k groups, group i ends at sorted
    position floor(P * (n - 1) / 100 + 1) for R7, or floor(P * (n + 1) / 100)
    for R6 (the last group ends at n), where P = 100 * i / k. Sales with the
    same proxy keep their input order; the Standard does not specify one.

    Further analysis (Appendix E.4): the group table includes each group's
    COD, and ``notes`` flags an inner group with the lowest or highest median
    ratio, which the Standard says to review.

    Sale prices and appraised values are paired by position, so numpy
    arrays, lists, and pandas columns (including columns of a sorted or
    filtered DataFrame) all give the same result.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.VEI(sales["Sale_Price"], sales["Appraised_Value"])
    >>> print(result)                         # plain-English report
    >>> result.vei, result.ci_lower, result.ci_upper
    >>> result.conclusion                     # e.g. 'Progressive'
    >>> result.groups                         # one row per value group
    >>> sales = sales.join(result.sales["Group"])   # each sale's value group
    """
    ranking_method = _check_vei_options(confidence_level, ranking_method)

    # Check the data (see "Data checks" in the module documentation). Sales
    # with an appraised value of 0 or less are not kept here, as they are for
    # the other statistics: the VEI cannot use them, so they are left out.
    sale_price, appraised_value, row_labels = _sales_pairs(
        Sale_Amount, Appr_Value, return_labels=True, report_unusual_appraised=False
    )
    sale_price, appraised_value, row_labels = _vei_leave_out_ratios_of_zero_or_less(
        sale_price, appraised_value, row_labels
    )

    result = _vei_calculation(sale_price, appraised_value, row_labels, confidence_level, ranking_method)
    if show_plot:
        result.plot()
    return result


def VEI_Lower(
    Sale_Amount: np.ndarray,
    Appr_Value: np.ndarray,
    confidence_level: float = 0.90,
    ranking_method: str = "R7",
) -> float:
    """
    Lower limit of the VEI's confidence interval, in percent.

    See ``VEI`` for the calculation and the meaning of each argument.

    Parameters
    ----------
    Sale_Amount : array-like
        Sale prices (numpy array, list, or pandas Series).
    Appr_Value : array-like
        Appraised / model-predicted values, in the same order as
        ``Sale_Amount``.
    confidence_level : float, default 0.90
        Confidence level of the interval.
    ranking_method : {"R7", "R6"}, default "R7"
        How sales are assigned to groups.

    Returns
    -------
    float
        The lower confidence limit. NaN when there are fewer than 20 sales.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.VEI_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    return VEI(Sale_Amount, Appr_Value, confidence_level, ranking_method).ci_lower


def VEI_Upper(
    Sale_Amount: np.ndarray,
    Appr_Value: np.ndarray,
    confidence_level: float = 0.90,
    ranking_method: str = "R7",
) -> float:
    """
    Upper limit of the VEI's confidence interval, in percent.

    See ``VEI`` for the calculation and the meaning of each argument.

    Parameters
    ----------
    Sale_Amount : array-like
        Sale prices (numpy array, list, or pandas Series).
    Appr_Value : array-like
        Appraised / model-predicted values, in the same order as
        ``Sale_Amount``.
    confidence_level : float, default 0.90
        Confidence level of the interval.
    ranking_method : {"R7", "R6"}, default "R7"
        How sales are assigned to groups.

    Returns
    -------
    float
        The upper confidence limit. NaN when there are fewer than 20 sales.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.VEI_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    return VEI(Sale_Amount, Appr_Value, confidence_level, ranking_method).ci_upper


def VEI_Conclusion(
    Sale_Amount: np.ndarray,
    Appr_Value: np.ndarray,
    confidence_level: float = 0.90,
    ranking_method: str = "R7",
) -> str:
    """
    The VEI's conclusion: regressive, progressive, or neither.

    See ``VEI`` for the calculation, the meaning of each argument, and the
    rules (Table E.7.1 of the Standard).

    Parameters
    ----------
    Sale_Amount : array-like
        Sale prices (numpy array, list, or pandas Series).
    Appr_Value : array-like
        Appraised / model-predicted values, in the same order as
        ``Sale_Amount``.
    confidence_level : float, default 0.90
        Confidence level of the interval.
    ranking_method : {"R7", "R6"}, default "R7"
        How sales are assigned to groups.

    Returns
    -------
    str
        ``'Regressive'`` (the whole confidence interval is below -10%),
        ``'Progressive'`` (the whole interval is above +10%),
        ``'No significant progressivity or regressivity'``, or
        ``'Not enough sales (fewer than 20)'``.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.VEI_Conclusion(sales["Sale_Price"], sales["Appraised_Value"])
    'No significant progressivity or regressivity'
    """
    return VEI(Sale_Amount, Appr_Value, confidence_level, ranking_method).conclusion
