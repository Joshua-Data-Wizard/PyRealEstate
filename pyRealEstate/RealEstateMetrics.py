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
  rows.
* Inputs of different lengths, empty inputs, inputs that are not numbers, or
  inputs with no usable sales raise a ``ValueError``.

Rows are identified by the pandas index when a pandas Series is given, and
otherwise by position (starting at 0).
"""

import math
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Optional, Tuple


class DataQualityWarning(UserWarning):
    """
    Warning about sales that were left out or that look unusual.

    These warnings can be silenced or caught on their own, for example::

        import warnings
        from pyRealEstate.RealEstateMetrics import DataQualityWarning

        warnings.simplefilter("ignore", DataQualityWarning)
    """


# How many rows to list in a data-quality message before summarizing the rest.
_MAX_ROWS_LISTED = 10


def _warn_data_quality(message: str) -> None:
    """Issue a ``DataQualityWarning`` (every data-quality warning goes through here)."""
    warnings.warn(message, DataQualityWarning, stacklevel=1)


def _reissue_once(caught: List[warnings.WarningMessage]) -> None:
    """
    Re-issue collected warnings, reporting each distinct data-quality problem once.

    Used by ``DOR_SUMMARY_Statistics``: every statistic it calculates checks the
    same data, so each problem would otherwise be reported once per statistic.
    Any other warnings are passed on unchanged.
    """
    reported = set()
    for caught_warning in caught:
        if issubclass(caught_warning.category, DataQualityWarning):
            message = str(caught_warning.message)
            if message not in reported:
                reported.add(message)
                _warn_data_quality(message)
        else:
            warnings.warn_explicit(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
            )


def _describe_rows(labels: np.ndarray) -> str:
    """Format row labels for a message, e.g. 'rows 5, 17, 22'."""
    shown = ", ".join(str(label) for label in labels[:_MAX_ROWS_LISTED])
    if len(labels) > _MAX_ROWS_LISTED:
        shown += f" and {len(labels) - _MAX_ROWS_LISTED} more"
    return ("row " if len(labels) == 1 else "rows ") + shown


def _problem_lines(problems: List[Tuple[str, np.ndarray]], row_labels: np.ndarray) -> str:
    """One line per problem that occurs, e.g. '  - missing sale price: 2 (rows 5, 17)'."""
    return "\n".join(
        f"  - {description}: {int(mask.sum())} ({_describe_rows(row_labels[mask])})"
        for description, mask in problems
        if mask.any()
    )


def _sales_pairs(y: Any, x: Any) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    sale_price, appraised_value : numpy.ndarray
        The usable sales, paired by position. pandas decides the number type,
        so for example a list of whole numbers becomes int64 on every platform.

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
    if isinstance(y, pd.Series):
        row_labels = y.index.to_numpy()
    elif isinstance(x, pd.Series):
        row_labels = x.index.to_numpy()
    else:
        row_labels = np.arange(len(sale_price))

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
    if any(mask.any() for _, mask in unusual):
        _warn_data_quality(
            "pyRealEstate found unusual appraised values. These sales are included "
            "in the statistics; please check that they are correct:\n"
            + _problem_lines(unusual, row_labels)
        )

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


def VEI (Sale_Amount : np.ndarray, Appr_Value : np.ndarray  , confidence_level = 0.90 , Sales_Ratio_Accuracy = 3 , show_plot = False):
    """
    Vertical Equity Indicator (VEI).

    VEI checks vertical equity by comparing the median sale ratio of the
    lowest-value group of properties with that of the highest-value group.
    If higher-value properties have systematically lower (or higher) sale
    ratios than lower-value properties, the appraisals are regressive (or
    progressive).

    Parameters
    ----------
    Sale_Amount : array-like
        Sale prices (numpy array, list, or pandas Series).
    Appr_Value : array-like
        Appraised / model-predicted values, in the same order as
        ``Sale_Amount``.
    confidence_level : float, default 0.90
        Confidence level of the interval calculated around each group's
        median sale ratio.
    Sales_Ratio_Accuracy : int, default 3
        Number of decimal places the sale ratios are rounded to before the
        group medians and confidence intervals are calculated.
    show_plot : bool, default False
        If True, show a box plot of sale ratios by market-value group, with
        dashed lines at the two confidence bounds being compared.

    Returns
    -------
    Working_VEI_df : pandas.DataFrame
        One row per sale, sorted by ``Market_Value_Proxy``, with the columns
        ``Appr_Value``, ``Sale_Amount``, ``Raw_Sale_Ratio``, ``Sale_Ratios``
        (rounded), ``Market_Value_Proxy`` and ``Market_Value_Group``.
    Group_VEI_Data : pandas.DataFrame
        One row per market-value group, with the columns ``Group``,
        ``Group_Count``, ``Group_Median`` (median market-value proxy),
        ``Group Median_Sale_Ratio``, and ``Group_Lower_Bound`` /
        ``Group_Upper_Bound`` (the confidence interval of the group's median
        sale ratio).
    VEI_Score : float
        100 * (median ratio of the highest-value group - median ratio of the
        lowest-value group) / overall median ratio. A negative score means
        higher-value properties have lower sale ratios (regressivity).
    VEI_Sig : float
        0 if the confidence intervals of the lowest- and highest-value groups
        overlap. Otherwise, the gap between the two intervals as a percentage
        of the overall median ratio (always positive).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if a
        market-value group has too few sales to calculate the confidence
        interval of its median at ``confidence_level``.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Notes
    -----
    The procedure:

    1. Sale ratio = appraised value / sale price, rounded to
       ``Sales_Ratio_Accuracy`` decimal places.
    2. Market-value proxy = 0.5 * sale price + 0.5 * (appraised value /
       median ratio), rounded to whole dollars. Blending both values avoids
       grouping on either one alone.
    3. Sales are sorted by the proxy and split into groups of (nearly) equal
       size: 10 groups when there are more than 500 sales, 4 groups for
       51-500 sales, and 2 groups for 50 or fewer.
    4. Within each group of n sales, the confidence interval of the median
       sale ratio is found by rank. Let z be the two-sided normal critical
       value for ``confidence_level`` and j = z * sqrt(n) / 2, plus 0.5 when
       n is even, rounded up. With the group's ratios sorted and positions
       numbered from 0, the median position is m = (n - 1) / 2. The lower
       bound is the ratio at position ceil(m - j). The upper bound is at
       position ceil(m + j) when n is odd, or floor(m + j) when n is even.
    5. ``VEI_Score`` and ``VEI_Sig`` compare the lowest- and highest-value
       groups as described under Returns. The overall median ratio is the
       median of the rounded sale ratios.

    Sale prices and appraised values are paired by position, so numpy
    arrays, lists, and pandas columns (including columns of a sorted or
    filtered DataFrame) all give the same result.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> working, groups, vei_score, vei_sig = rem.VEI(
    ...     sales["Sale_Price"], sales["Appraised_Value"]
    ... )
    """
    # Check the data and pair each sale price with its appraised value by
    # position (see "Data checks" in the module documentation). pandas (not
    # numpy) decides the column types, as it always has here, so a list of
    # whole numbers becomes int64 on every platform.
    Sale_Amount, Appr_Value = _sales_pairs(Sale_Amount, Appr_Value)

    Sale_Ratio_Raw = Appr_Value / Sale_Amount
    Sale_Ratios = np.round(Sale_Ratio_Raw, Sales_Ratio_Accuracy)

    # Rows stay in input order until the sort below. VEI depends on this:
    # tied Market_Value_Proxy values are common, and the sort is not stable.
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

        lower_index = math.ceil(group_median_ratio_index - Temp_Z_Value)
        if group_data.shape[0] % 2 == 0:
            upper_index = math.floor(group_median_ratio_index + Temp_Z_Value)
        else:
            upper_index = math.ceil(group_median_ratio_index + Temp_Z_Value)

        # Both confidence-interval positions must fall inside the group.
        if lower_index < 0 or upper_index > group_n - 1:
            raise ValueError(
                f"VEI market-value group {i} has only {group_n} sales, which is "
                f"too few to calculate a {confidence_level:.0%} confidence "
                "interval for its median. Use more sales or a lower "
                "confidence_level."
            )

        lower_bound = group_data['Sale_Ratios'].iloc[lower_index]
        upper_bound = group_data['Sale_Ratios'].iloc[upper_index]
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