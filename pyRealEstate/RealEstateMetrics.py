"""
Ratio-study statistics for mass appraisal: weighted mean sale ratio, COD,
PRD, PRB, VEI, and the Kruskal-Wallis test.

Argument convention used by every function in this module::

    y  (VEI: Sale_Amount) : sale prices
    x  (VEI: Appr_Value)  : appraised / model-predicted values
    sale ratio            = appraised value / sale price
    groups (Kruskal_Wallis only) : the group of each sale

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
* ``Kruskal_Wallis`` also leaves out sales without a group label, and reports
  them.
* Inputs of different lengths, empty inputs, inputs that are not numbers, or
  inputs with no usable sales raise a ``ValueError``.

Rows are identified by the pandas index when a pandas Series is given, and
otherwise by position (starting at 0).
"""

import math
import numbers
import textwrap
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels, scipy, and matplotlib are imported inside the functions that
# use them. Each takes up to ~1.5 s to import, and users who only need the
# weighted mean, COD, or PRD should not have to wait for them.
if TYPE_CHECKING:  # for editors and type checkers only; never imported at run time
    import matplotlib.axes

from ._data_checks import (
    DataQualityWarning,
    _problem_lines,
    _reissue_once,
    _row_labels,
    _warn_data_quality,
)
from ._plotting import (
    _MINUS,
    _PLOT_STYLE,
    _add_title,
    _compact_dollars,
    _fit_title,
    _new_axes,
    _number,
    _plot_width_inches,
    _style_axes,
)


def _sales_pairs(
    y: Any,
    x: Any,
    return_rows: bool = False,
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
    return_rows : bool, default False
        Also return which rows were kept: the row label of each usable sale
        (the pandas index, or the position), so the caller can name rows or
        label its results, and a True/False mask over the input positions, so
        the caller can filter other values that belong to each sale (such as
        group labels) in exactly the same way.
    report_unusual_appraised : bool, default True
        Warn about appraised values of 0 or less (which are kept). ``VEI``
        turns this off because it leaves those sales out and reports them
        itself.

    Returns
    -------
    sale_price, appraised_value : numpy.ndarray
        The usable sales, paired by position. pandas decides the number type,
        so for example a list of whole numbers becomes int64 on every platform.
    row_labels, kept : numpy.ndarray
        Only when ``return_rows`` is True: the row label of each usable sale,
        and a boolean mask, one entry per input position, True where the sale
        was kept.

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

    if return_rows:
        return sale_price, appraised_value, row_labels, keep
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

    Calculates the weighted mean sale ratio, COD, PRD, PRB (with its 95%
    confidence interval and PASS / FAIL conclusion), and the Vertical Equity
    Indicator (VEI, with its 90% confidence interval and conclusion), prints
    them, and flags COD and PRD values that fall outside common target
    ranges. The VEI is the Standard's preferred measure of vertical equity
    (Standard on Ratio Studies 2026, Section 8.2.1).

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
        ``'PRB_Upper'``, ``'PRB_Conclusion'``, ``'VEI'``, ``'VEI_Lower'``,
        ``'VEI_Upper'`` and ``'VEI_Conclusion'`` (the VEI values are in
        percent, and NaN with fewer than 20 sales). Otherwise ``None``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the VEI (it needs 20). Each statistic checks the data, but each
        problem is reported once.

    Notes
    -----
    Each statistic is calculated by the function of the same name in this
    module (``weighted_Mean_Sale_Ratio``, ``COD``, ``PRD``, ``PRB``,
    ``PRB_Lower``, ``PRB_Upper``, ``PRB_Conclusion``, ``VEI``). See their
    documentation for the formulas. The VEI uses its defaults (a 90%
    confidence interval and R7 ranking, as in the Standard's worked
    example); call ``VEI`` directly for other options, its group table, or
    its full report. Like ``VEI``, the summary's VEI leaves out sales with an
    appraised value of 0 or less, which the other statistics keep.

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
    >>> results["COD"], results["VEI_Conclusion"]
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
        vei = VEI(y, x)
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

    if vei.scenario is None:  # fewer than 20 usable sales
        print(f"\nVEI: not calculated <-> VEI RESULT: {vei.conclusion}")
    else:
        print(
            f"\nVEI: {vei.vei:.2f}% <-> "
            f"VEI {_percent(vei.confidence_level)} CI: {vei.ci_lower:.2f}% to {vei.ci_upper:.2f}% <-> "
            f"VEI RESULT: {vei.conclusion}"
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
            'VEI': vei.vei,
            'VEI_Lower': vei.ci_lower,
            'VEI_Upper': vei.ci_upper,
            'VEI_Conclusion': vei.conclusion,
        }
    return None


# ---------------------------------------------------------------------------
# Small helpers shared by the reports and option checks below
# ---------------------------------------------------------------------------

def _percent(fraction: float) -> str:
    """0.9 -> '90%', 0.975 -> '97.5%'."""
    return f"{fraction * 100:g}%"


def _format_p_value(p_value: float) -> str:
    """'= 0.0273' for reports, or '< 0.0001' when the p-value is smaller than that."""
    return "< 0.0001" if p_value < 0.0001 else f"= {p_value:.4f}"


def _check_fraction(value: float, name: str, example: str) -> None:
    """Raise a ValueError unless ``value`` is a number strictly between 0 and 1."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not 0 < value < 1  # also catches NaN
    ):
        raise ValueError(f"{name} must be a number between 0 and 1 ({example}), not {value!r}.")


# ---------------------------------------------------------------------------
# Kruskal-Wallis test
#
# Source: IAAO Standard on Ratio Studies (2026): Section 6.4 (Table 6.3,
# "three or more property groups are appraised at equal percentages of market
# value"), Appendix E.4 (the VEI's value groups) and Appendix F (Table F.1.1).
# ---------------------------------------------------------------------------

_KW_SMALL_GROUP = 5  # groups with fewer sales get a warning (user decision, 2026-10-26)

# The conclusions returned by Kruskal_Wallis
_KW_SIGNIFICANT = "Significant difference between groups"
_KW_NOT_SIGNIFICANT = "No significant difference between groups"
_KW_NOT_ENOUGH_GROUPS = "Not enough groups (fewer than 2)"


@dataclass(frozen=True, eq=False, repr=False)
class KruskalWallisResult:
    """
    The result of ``Kruskal_Wallis``: are the groups appraised at the same
    level?

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    statistic : float
        The Kruskal-Wallis statistic H, corrected for tied ratios. The more
        the groups' average ranks differ, the larger it is. NaN when it
        cannot be calculated: fewer than 2 groups, or every ratio identical.
    p_value : float
        The chance of seeing differences at least this large if every group
        were appraised at the same level. NaN when ``statistic`` is NaN.
    degrees_of_freedom : int
        The number of groups minus 1 (0 when there are fewer than 2 groups).
    significance_level : float
        The significance level used for the conclusion, e.g. 0.05.
    conclusion : str
        ``'Significant difference between groups'`` (the p-value is below the
        significance level), ``'No significant difference between groups'``,
        or ``'Not enough groups (fewer than 2)'``.
    groups : pandas.DataFrame
        One row per group, in sorted order of the labels (or in order of
        first appearance when the labels cannot be sorted):

        * ``Group``: the group label;
        * ``Sales``: the number of sales in the group;
        * ``Median_Ratio``: the group's median sale ratio;
        * ``Mean_Rank``: the average rank of the group's ratios among all
          the ratios (1 = the lowest ratio).
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    significance_level: float
    conclusion: str
    groups: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"KruskalWallisResult(statistic={self.statistic:.4f}, p_value={self.p_value:.4f}, "
            f"degrees_of_freedom={self.degrees_of_freedom}, conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        lines = [
            "Kruskal-Wallis test: are the groups appraised at the same level?",
            "(IAAO Standard on Ratio Studies, 2026: Section 6.4 and Appendix F)",
            f"{int(self.groups['Sales'].sum())} sales in {len(self.groups)} groups",
        ]
        if self.conclusion == _KW_NOT_ENOUGH_GROUPS:
            lines.append(f"Conclusion: {self.conclusion}")
            return "\n".join(lines)

        group_table = self.groups.to_string(
            index=False,
            formatters={"Median_Ratio": "{:.4f}".format, "Mean_Rank": "{:.2f}".format},
        )
        lines += [
            self._statistic_text(),
            f"Conclusion at the {_percent(self.significance_level)} significance level: "
            f"{self.conclusion}",
            "",
            group_table,
        ]
        return "\n".join(lines)

    def _statistic_text(self) -> str:
        """'H = 7.2000 (2 degrees of freedom), p-value = 0.0273', or why there is no H."""
        if math.isnan(self.statistic):
            return "H cannot be calculated: every sale ratio is the same, so there is nothing to rank."
        degrees = "degree" if self.degrees_of_freedom == 1 else "degrees"
        return (
            f"H = {self.statistic:.4f} ({self.degrees_of_freedom} {degrees} of freedom), "
            f"p-value {_format_p_value(self.p_value)}"
        )


def _group_order(labels: np.ndarray) -> list:
    """Each group label once: sorted when possible, otherwise in order of first appearance."""
    unique_labels = list(pd.unique(labels))
    try:
        return sorted(unique_labels)
    except TypeError:  # e.g. a mix of numbers and text, which Python cannot sort
        return unique_labels


def _kruskal_wallis_calculation(
    ratio: np.ndarray, group_labels: np.ndarray, significance_level: float
) -> KruskalWallisResult:
    """The Kruskal-Wallis test on sales that have passed the data checks."""
    n_sales = len(ratio)
    order = _group_order(group_labels)
    n_groups = len(order)

    # Rank every ratio, 1 for the lowest. Tied ratios share the average of their ranks.
    ratios = pd.Series(ratio, dtype=float)
    ranks = ratios.rank(method="average")
    by_group = pd.DataFrame({"Ratio": ratios, "Rank": ranks}).groupby(group_labels, sort=False)
    groups = pd.DataFrame({
        "Group": order,
        "Sales": by_group.size().reindex(order).to_numpy(),
        "Median_Ratio": by_group["Ratio"].median().reindex(order).to_numpy(),
        "Mean_Rank": by_group["Rank"].mean().reindex(order).to_numpy(),
    })

    def empty_result(conclusion: str) -> KruskalWallisResult:
        return KruskalWallisResult(
            statistic=math.nan, p_value=math.nan, degrees_of_freedom=max(n_groups - 1, 0),
            significance_level=significance_level, conclusion=conclusion, groups=groups,
        )

    if n_groups < 2:
        _warn_data_quality(
            "The Kruskal-Wallis test needs at least 2 groups to compare, but the usable sales "
            f"are in {n_groups} group{'' if n_groups == 1 else 's'}. Its statistic and p-value "
            "are left empty (NaN)."
        )
        return empty_result(_KW_NOT_ENOUGH_GROUPS)

    small = groups[groups["Sales"] < _KW_SMALL_GROUP]
    if len(small):
        listed = [f"{row.Group} ({row.Sales} sale{'' if row.Sales == 1 else 's'})" for row in small.itertuples()]
        if len(listed) > 10:
            listed = listed[:10] + [f"and {len(listed) - 10} more"]
        _warn_data_quality(
            f"{len(small)} of {n_groups} groups {'has' if len(small) == 1 else 'have'} fewer than "
            f"{_KW_SMALL_GROUP} sales: {', '.join(listed)}. The Kruskal-Wallis test still runs, but "
            "its p-value is less reliable for small groups (the Standard, Table F.1.1: groups need "
            "an adequate sample size)."
        )

    # Tie correction, for each run of t equal ratios:  C = 1 - sum(t^3 - t) / (N^3 - N)
    tie_sizes = ratios.value_counts().to_numpy(dtype=float)
    tie_correction = 1 - (tie_sizes ** 3 - tie_sizes).sum() / (float(n_sales) ** 3 - n_sales)
    if tie_correction == 0:  # every ratio is identical: there is nothing to rank
        _warn_data_quality(
            f"Every sale ratio is the same ({ratios.iloc[0]:.4f}), so the Kruskal-Wallis statistic "
            "cannot be calculated: there is nothing to rank. The groups do not differ. Identical "
            "ratios are unusual in real data; please check them."
        )
        return empty_result(_KW_NOT_SIGNIFICANT)

    # H measures how far each group's average rank is from the overall average
    # rank, (N + 1) / 2:
    #     H = 12 / (N (N + 1)) * sum over groups of  n_i * (mean rank_i - (N + 1) / 2)^2
    # then divided by the tie correction. (The textbook shortcut
    # 12/(N(N+1)) * sum(R_i^2 / n_i) - 3(N + 1) is the same number, but it
    # subtracts two large, nearly equal numbers and loses precision.)
    overall_mean_rank = (n_sales + 1) / 2
    spread = (groups["Sales"] * (groups["Mean_Rank"] - overall_mean_rank) ** 2).sum()
    statistic = 12 / (n_sales * (n_sales + 1)) * spread / tie_correction

    # If the groups do not differ, H follows a chi-square distribution with
    # (groups - 1) degrees of freedom; the p-value is its upper tail beyond H.
    from scipy import stats  # imported here: slow to import (see top of module)

    p_value = float(stats.chi2.sf(statistic, n_groups - 1))
    conclusion = _KW_SIGNIFICANT if p_value < significance_level else _KW_NOT_SIGNIFICANT
    return KruskalWallisResult(
        statistic=float(statistic), p_value=p_value, degrees_of_freedom=n_groups - 1,
        significance_level=significance_level, conclusion=conclusion, groups=groups,
    )


def Kruskal_Wallis(
    y: np.ndarray,
    x: np.ndarray,
    groups: Any,
    significance_level: float = 0.05,
) -> KruskalWallisResult:
    """
    Kruskal-Wallis test: are the groups appraised at the same level?

    Tests whether sales in different groups (neighborhoods, property classes,
    value groups, ...) are appraised at the same percentage of market value.
    Every sale ratio is ranked from lowest to highest; if one group's ratios
    sit systematically higher or lower in that ranking than another's, the
    groups are not appraised at the same level. The test uses ranks, not the
    ratios themselves, so it does not assume the ratios follow a normal
    distribution, and outliers have little influence.

    The IAAO Standard on Ratio Studies (2026) lists it in Section 6.4,
    Table 6.3, as the test that "three or more property groups are appraised
    at equal percentages of market value", and in Appendix E.4 and
    Appendix F (Table F.1.1) to compare the value groups of a vertical
    equity study. With two groups it is the same test as the two-sided
    Mann-Whitney test.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    groups : array-like
        The group of each sale (e.g. a neighborhood code), in the same order
        as ``y``. Any labels work: text, numbers, or a pandas categorical.
    significance_level : float, default 0.05
        The groups are said to differ significantly when the p-value is below
        this level. The Standard's example uses 95% confidence, i.e. 0.05.

    Returns
    -------
    KruskalWallisResult
        The statistic H (``statistic``), its ``p_value``, the
        ``degrees_of_freedom``, the ``conclusion``, and a table with each
        group's sales, median ratio and average rank (``groups``).
        ``print(result)`` shows a plain-English report.

    Raises
    ------
    ValueError
        If the inputs differ in length (including ``groups``), are empty, are
        not numbers, or have no usable sales; or if ``significance_level`` is
        not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out (see "Data checks" in the module documentation),
        including sales without a group label; if a group has fewer than 5
        sales; if fewer than 2 groups remain; or if every ratio is identical.

    See Also
    --------
    VEI : with ``kruskal_wallis=True``, runs this test on the VEI's value
        groups.

    Notes
    -----
    The calculation::

        ratio      = appraised value / sale price
        rank       = position of each ratio among all N ratios, lowest = 1
                     (tied ratios share the average of their positions)
        mean_i     = average rank of the n_i ratios in group i
        H_raw      = 12 / (N (N + 1)) * sum over groups of n_i * (mean_i - (N + 1) / 2)^2
        C          = 1 - sum over each run of t tied ratios of (t^3 - t) / (N^3 - N)
        H          = H_raw / C
        p-value    = chance that a chi-square variable with (groups - 1)
                     degrees of freedom exceeds H

    (N + 1) / 2 is the average rank over all sales, so H grows as the
    groups' average ranks move apart. The conclusion is "Significant
    difference between groups" when the p-value is below
    ``significance_level``.

    Special cases, which give NaN rather than an error so that a loop over
    many strata keeps running:

    * Fewer than 2 groups: conclusion ``'Not enough groups (fewer than 2)'``.
    * Every ratio identical: H cannot be calculated (there is nothing to
      rank), and the conclusion is ``'No significant difference between
      groups'``.

    Small groups make the p-value less reliable; the Standard asks for "an
    adequate sample size" (Table F.1.1) without a number. A group with fewer
    than 5 sales therefore gets a warning, a common rule of thumb for this
    test, but the test still runs.

    Group labels are paired with the sales by position, like the sale prices
    and appraised values. Sales without a group label (None, NaN) are left
    out, with a warning. Appraised values of 0 or less are kept, with the
    usual warning (ranks need no logarithm).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Kruskal_Wallis(
    ...     sales["Sale_Price"], sales["Appraised_Value"], sales["Neighborhood"]
    ... )
    >>> print(result)                     # plain-English report
    >>> result.p_value, result.conclusion
    >>> result.groups                     # each group's sales, median ratio, average rank
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")

    group_labels = pd.Series(groups).to_numpy()  # paired by position; categoricals become plain labels
    if len(group_labels) != len(y):
        raise ValueError(
            f"There are {len(y)} sale prices (y) but {len(group_labels)} group labels; each sale "
            "needs exactly one group label."
        )

    # Check the sales (see "Data checks" in the module documentation), and keep
    # the group labels of the same sales.
    sale_price, appraised_value, row_labels, kept = _sales_pairs(y, x, return_rows=True)
    group_labels = group_labels[kept]

    # A sale without a group cannot be compared with anything: leave it out.
    no_group = np.asarray(pd.isna(group_labels), dtype=bool)
    if no_group.any():
        _warn_data_quality(
            f"The Kruskal-Wallis test left out {int(no_group.sum())} of {len(no_group)} sales that "
            "have no group:\n"
            + _problem_lines([("missing group label", no_group)], row_labels)
            + f"\nThe test uses the remaining {int((~no_group).sum())} sales."
        )
        has_group = ~no_group
        sale_price, appraised_value = sale_price[has_group], appraised_value[has_group]
        group_labels = group_labels[has_group]

    ratio = appraised_value / sale_price
    return _kruskal_wallis_calculation(ratio, group_labels, significance_level)


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
    charts the test and the value groups (see ``plot``). The values are
    read-only.

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
    kruskal_wallis : KruskalWallisResult or None
        The Kruskal-Wallis test across all the value groups (Appendix E.4),
        when ``VEI`` was called with ``kruskal_wallis=True``; otherwise None.
        Also None when there are fewer than 20 sales (there are no groups).
        Information only: it never changes the VEI's conclusion.
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
    kruskal_wallis: Optional[KruskalWallisResult] = None

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
        if self.kruskal_wallis is not None:
            test = self.kruskal_wallis
            lines += [
                "",
                textwrap.fill(
                    f"Kruskal-Wallis test across all {len(self.groups)} groups (information only; it "
                    f"does not change the VEI conclusion): {test._statistic_text()}. "
                    f"{test.conclusion} at the {_percent(test.significance_level)} significance level.",
                    width=88,
                ),
                textwrap.fill(
                    "Caution (Appendix E.4): keep the statistical considerations of multiple "
                    "comparisons in mind, and note that samples with higher dispersion can mask "
                    "differences, especially when they are small.",
                    width=88,
                ),
            ]
        return "\n".join(lines)

    def plot(
        self,
        kind: str = "summary",
        title: Optional[str] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        show: bool = True,
    ) -> None:
        """
        Chart the result: the VEI test, the sale ratios of each value group,
        or both.

        Parameters
        ----------
        kind : {"summary", "test", "groups"}, default "summary"
            Which chart to draw:

            * ``"test"``: the VEI (a dot) and its confidence interval (a bar)
              on a number line, with the acceptable range of -10% to +10%
              shaded. Whether the bar clears the shaded range *is* the
              Standard's test (Table E.7.1), so the conclusion shows at a
              glance.
            * ``"groups"``: a box plot of the sale ratios in each value
              group, lowest values on the left (Appendix E.4). The first and
              last groups, the two the VEI compares, are blue, with their
              median ratios labeled, and a line marks the overall median
              ratio. Under each box are the group's median market value
              proxy, number of sales, and COD. An inner group with the
              lowest or highest median ratio is marked "review (E.4)", as
              the Standard recommends.
            * ``"summary"``: both in one figure, ``"test"`` above
              ``"groups"``.
        title : str, optional
            Your own title instead of the default ("Vertical equity (VEI):
            <conclusion>" for ``"summary"`` and ``"test"``, "Sale ratios by
            value group" for ``"groups"``). The subtitle then starts with the
            VEI and its conclusion, so the chart still states them.
        ax : matplotlib.axes.Axes, optional
            Draw on these axes (e.g. one panel of your own figure) instead of
            a new figure, leaving your figure's colors alone. Only for
            ``"test"`` and ``"groups"``: ``"summary"`` has two panels and
            always makes its own figure.
        show : bool, default True
            Display the chart. Use False to keep customizing it or to save
            it, for example with ``plt.savefig("vei.png", dpi=300)``.

        Returns
        -------
        None
            The chart is drawn on a new figure, or on ``ax``.
            ``plt.gcf()`` returns the figure.

        Raises
        ------
        ValueError
            If ``kind`` is not "summary", "test" or "groups", or if ``ax`` is
            given with ``kind="summary"``.

        Notes
        -----
        Left of 0% is regressive and right of 0% is progressive (Section 8.2:
        a positive VEI means the higher-value properties have the higher
        sale ratios). The "test" chart labels both directions.

        When ``VEI`` was run with ``kruskal_wallis=True``, the box plot's
        subtitle includes the test's p-value (information only; it never
        changes the conclusion).

        With fewer than 20 sales there is no VEI to draw, so every kind shows
        only its title and the reason.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from pyRealEstate import RealEstateMetrics as rem
        >>> result = rem.VEI(sales["Sale_Price"], sales["Appraised_Value"])
        >>> result.plot()                                   # both charts
        >>> result.plot(kind="test", title="Residential, 2026 roll")
        >>> result.plot(kind="groups", show=False)
        >>> plt.savefig("vei_groups.png", dpi=300)
        >>> fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6))
        >>> result.plot(kind="groups", ax=left, show=False)  # one panel of your figure
        """
        import matplotlib.pyplot as plt  # imported here: slow to import (see top of module)

        if kind not in _VEI_PLOT_KINDS:
            raise ValueError(f"kind must be 'summary', 'test' or 'groups', not {kind!r}.")
        if kind == "summary" and ax is not None:
            raise ValueError(
                "kind='summary' draws two panels in its own figure, so it cannot use ax. To draw "
                "on your own axes, use kind='test' or kind='groups'."
            )

        if self.scenario is None:  # fewer than 20 sales: there is no VEI to draw
            _vei_plot_not_enough_sales(self, title, ax)
        elif kind == "summary":
            _vei_plot_summary(self, title)
        elif kind == "test":
            _vei_plot_test(self, title, ax)
        else:
            _vei_plot_groups(self, title, ax)
        if show:
            plt.show()


# ---------------------------------------------------------------------------
# VEI charts (VEIResult.plot). The shared colors and layout are in _plotting.py.
# ---------------------------------------------------------------------------

_VEI_PLOT_KINDS = ("summary", "test", "groups")


def _vei_ci_facts(result: VEIResult) -> List[str]:
    """
    The confidence interval and which row of Table E.7.1 applies, as parts
    of a subtitle, e.g. "90% CI 29.52% to 67.54%, entirely above +10%
    (scenario 4)".
    """
    ci = (
        f"{_percent(result.confidence_level)} CI {_number(result.ci_lower)}% to "
        f"{_number(result.ci_upper)}%"
    )
    return {
        1: [ci, "VEI within ±10% (scenario 1)"],
        2: [f"{ci} overlaps ±10%, so not significant (scenario 2)"],
        3: [f"{ci}, entirely below {_MINUS}10% (scenario 3)"],
        4: [f"{ci}, entirely above +10% (scenario 4)"],
    }[result.scenario]


def _vei_verdict_fact(result: VEIResult) -> str:
    """E.g. "VEI 48.53%: Progressive"."""
    return f"VEI {_number(result.vei)}%: {result.conclusion}"


def _vei_kruskal_wallis_fact(result: VEIResult) -> Optional[str]:
    """The Kruskal-Wallis p-value for a subtitle, or None when the test was not run."""
    test = result.kruskal_wallis
    if test is None:
        return None
    if math.isnan(test.p_value):
        return "Kruskal-Wallis test not calculated (every sale ratio is the same)"
    return f"Kruskal-Wallis test across the {len(result.groups)} groups: p {_format_p_value(test.p_value)}"


def _vei_draw_test(result: VEIResult, ax: "matplotlib.axes.Axes") -> None:
    """
    Draw the VEI test (Table E.7.1): the VEI as a dot and its confidence
    interval as a bar, over the shaded acceptable range of -10% to +10%.
    """
    import matplotlib.ticker as mticker  # imported here: slow to import (see top of module)

    style = _PLOT_STYLE
    _style_axes(ax, grid_axis=None)  # the shaded range is the only reference line needed
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)

    # Centered on 0%, so the acceptable range sits in the middle; wide enough
    # for the whole interval and its labels.
    reach = 1.18 * max(15.0, abs(result.ci_lower), abs(result.ci_upper))
    ax.set_xlim(-reach, reach)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: "0%" if value == 0 else _number(value, "+.0f", "%"))
    )

    ax.axvspan(-_VEI_LIMIT, _VEI_LIMIT, color=style["band"], linewidth=0, zorder=0)
    ax.text(0, 0.95, "acceptable range ±10%", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=style["muted"])

    # The interval, then the VEI on top of it (a ring of background color
    # keeps the dot distinct from the bar).
    ax.plot([result.ci_lower, result.ci_upper], [0, 0], color=style["blue"], linewidth=3.2,
            solid_capstyle="round", zorder=3)
    ax.scatter([result.vei], [0], s=90, color=style["blue_dark"], edgecolor=style["surface"],
               linewidth=2, zorder=4)
    ax.annotate(f"VEI {_number(result.vei)}%", (result.vei, 0), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=style["text"])
    for limit, side in ((result.ci_lower, -1), (result.ci_upper, 1)):  # beside each end
        ax.annotate(f"{_number(limit)}%", (limit, 0), xytext=(8 * side, 0),
                    textcoords="offset points", ha="right" if side < 0 else "left", va="center",
                    fontsize=8.5, color=style["text_secondary"])

    # Which way is which (Section 8.2: a positive VEI is progressive)
    for x, align, label in (
        (0.01, "left", "← regressive\nhigher values under-appraised"),
        (0.99, "right", "progressive →\nhigher values over-appraised"),
    ):
        ax.text(x, 0.05, label, transform=ax.transAxes, ha=align, va="bottom",
                multialignment=align, fontsize=8.5, color=style["muted"], linespacing=1.3)


def _vei_draw_groups(result: VEIResult, ax: "matplotlib.axes.Axes", own_figure: bool) -> float:
    """
    Draw the box plot of the sale ratios in each value group (Appendix E.4),
    with its legend above the plot.

    Returns the height of the legend, in points, so the title can be placed
    above it.
    """
    # Imported here: slow to import (see top of module).
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    style = _PLOT_STYLE
    _style_axes(ax)
    groups = result.groups
    n_groups = len(groups)
    deciles = n_groups > 4  # 10 groups: narrower labels
    positions = np.arange(1, n_groups + 1)
    box_width = 0.6 if deciles else 0.5

    # Each group's sale ratios, lowest values first
    ratios = [group.to_numpy() for _, group in result.sales.groupby("Group", sort=True)["Sale_Ratio"]]
    boxes = ax.boxplot(
        ratios, positions=positions, widths=box_width, patch_artist=True,
        medianprops={"linewidth": 2.2},
        whiskerprops={"color": style["context"], "linewidth": 1},
        capprops={"color": style["context"], "linewidth": 1},
        flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "none",
                    "markeredgecolor": style["context"], "markeredgewidth": 0.8},
    )

    # The first and last groups, the two the VEI compares, in blue; the others in gray.
    compared_fill = mcolors.to_rgba(style["blue"], 0.18)
    other_fill = mcolors.to_rgba(style["context"], 0.22)
    for i, (box, median) in enumerate(zip(boxes["boxes"], boxes["medians"])):
        compared = i in (0, n_groups - 1)
        box.set_facecolor(compared_fill if compared else other_fill)
        box.set_edgecolor(style["blue"] if compared else style["context_dark"])
        box.set_linewidth(1)
        median.set_color(style["blue_dark"] if compared else style["context_dark"])
    ax.axhline(result.overall_median, color=style["context_dark"], linewidth=1, zorder=1)

    # Label the two compared medians on their outer sides. A patch of the
    # background color keeps the overall-median line from crossing the numbers.
    label_background = {"boxstyle": "round,pad=0.15", "facecolor": ax.get_facecolor(), "edgecolor": "none"}
    for i, side in ((0, -1), (n_groups - 1, 1)):
        median = groups["Median_Ratio"].iloc[i]
        ax.annotate(f"{median:.3f}", (positions[i] + side * box_width / 2, median), xytext=(4 * side, 0),
                    textcoords="offset points", ha="right" if side < 0 else "left", va="center",
                    fontsize=8.5, fontweight="bold", color=style["text"], bbox=label_background)
    # Widen the plot on both sides until those labels fit inside it, whatever
    # its width. Each label is about 0.46 inch (five bold digits and the gap).
    # Solved from: (room beside the box + margin) * inches per group = label
    # width, where inches per group = plot width / (groups + 2 * margin).
    label_inches = 0.46
    plot_inches = _plot_width_inches(ax, own_figure)
    room_beside_box = 0.5 - box_width / 2
    margin = max(0.2, (label_inches * n_groups - room_beside_box * plot_inches)
                 / max(plot_inches - 2 * label_inches, 0.5))
    ax.set_xlim(0.5 - margin, n_groups + 0.5 + margin)

    # Appendix E.4: note an inner group with the lowest or highest median ratio.
    extremes = _vei_inner_extremes(groups)
    for i, word in extremes:
        ax.annotate(f"{word} median\nreview (E.4)", (positions[i], ratios[i].max()), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                    color=style["muted"], linespacing=1.2)
    if extremes:  # room above the tallest box for the notes
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.12 * (top - bottom))

    # Under each box: the group, its median proxy, its number of sales, and its COD
    labels = []
    for row in groups.itertuples():
        name = f"{row.Group}" if deciles else f"Group {row.Group}"
        sales = f"n={row.Sales:,}" if deciles else f"{row.Sales:,} sales"
        labels.append(f"{name}\n{_compact_dollars(row.Median_Proxy)}\n{sales}\nCOD {row.COD:.1f}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.8 if deciles else 8.5)
    ax.set_ylabel("Sale ratio (appraised value ÷ sale price)", color=style["text_secondary"], fontsize=10)
    ax.set_xlabel("Value groups, lowest to highest market value proxy (median proxy · sales · COD)",
                  color=style["text_secondary"], fontsize=9.5)

    handles = [mpatches.Patch(facecolor=compared_fill, edgecolor=style["blue"],
                              label="First and last groups (the VEI compares these)")]
    if n_groups > 2:
        handles.append(mpatches.Patch(facecolor=other_fill, edgecolor=style["context_dark"],
                                      label="Other groups"))
    handles.append(mlines.Line2D([], [], color=style["context_dark"], linewidth=1,
                                 label=f"Overall median ratio ({result.overall_median:.3f})"))

    # One row of entries when it fits the plot's width, otherwise one entry
    # per row. At 8.5 pt, a label is about 4.3 points per character; each
    # entry adds about 19 points (its key and padding), and 12 between entries.
    one_row_width = sum(4.3 * len(handle.get_label()) + 19 for handle in handles) + 12 * (len(handles) - 1)
    one_row = one_row_width <= 72 * plot_inches
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.3,
              ncol=len(handles) if one_row else 1, frameon=False, fontsize=8.5,
              labelcolor=style["text_secondary"], handlelength=1.4, columnspacing=1.4)
    return 20 if one_row else 20 + 15 * (len(handles) - 1)


def _vei_plot_test(result: VEIResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]) -> None:
    """``VEIResult.plot(kind="test")``."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 2.6))
    _vei_draw_test(result, ax)
    title, subtitle = _fit_title(ax, own_figure, *_vei_test_title(result, title))
    _add_title(ax, title, subtitle)


def _vei_test_title(result: VEIResult, title: Optional[str]) -> Tuple[str, List[str]]:
    """The title and subtitle parts of the "test" chart (also the top of the "summary")."""
    facts = _vei_ci_facts(result) + [f"{len(result.sales):,} sales in {len(result.groups)} value groups"]
    if title is None:
        return f"Vertical equity (VEI): {result.conclusion}", facts
    return title, [_vei_verdict_fact(result)] + facts


def _vei_plot_groups(result: VEIResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]) -> None:
    """``VEIResult.plot(kind="groups")``."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 5.8))
    legend_height = _vei_draw_groups(result, ax, own_figure)
    facts = [_vei_verdict_fact(result)] + _vei_ci_facts(result)
    kruskal_wallis = _vei_kruskal_wallis_fact(result)
    if kruskal_wallis is not None:
        facts.append(kruskal_wallis)
    title, subtitle = _fit_title(ax, own_figure, "Sale ratios by value group" if title is None else title, facts)
    _add_title(ax, title, subtitle, offset=4 + legend_height)


def _vei_plot_summary(result: VEIResult, title: Optional[str]) -> None:
    """``VEIResult.plot(kind="summary")``: the test above the box plot, in one figure."""
    import matplotlib.pyplot as plt  # imported here: slow to import (see top of module)

    figure = plt.figure(figsize=(9.5, 7.8), layout="constrained")
    figure.set_facecolor(_PLOT_STYLE["surface"])
    test_ax, groups_ax = figure.subplots(2, 1, height_ratios=[1, 3.1])
    for ax in (test_ax, groups_ax):
        ax.set_facecolor(_PLOT_STYLE["surface"])

    _vei_draw_test(result, test_ax)
    title, subtitle = _fit_title(test_ax, True, *_vei_test_title(result, title))
    _add_title(test_ax, title, subtitle)

    legend_height = _vei_draw_groups(result, groups_ax, own_figure=True)
    _add_title(groups_ax, "Sale ratios by value group", _vei_kruskal_wallis_fact(result),
               offset=4 + legend_height, title_size=10.5)


def _vei_plot_not_enough_sales(
    result: VEIResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]
) -> None:
    """With fewer than 20 sales there is no VEI to draw: only the title and the reason."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 1.0))
    ax.axis("off")
    reason = (
        f"{len(result.sales)} usable sales; the VEI needs at least {_VEI_MINIMUM_SALES} "
        "(Table E.3.1)"
    )
    if title is None:
        title, facts = f"Vertical equity (VEI): {result.conclusion}", [reason]
    else:
        facts = [result.conclusion, reason]
    title, subtitle = _fit_title(ax, own_figure, title, facts)
    _add_title(ax, title, subtitle)


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
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
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


def _vei_inner_extremes(groups: pd.DataFrame) -> List[Tuple[int, str]]:
    """
    Appendix E.4: "If another quantile group has the highest or lowest median
    ratio, it should be noted and reviewed to identify any correctable issues."

    Returns ``(row position, "lowest" or "highest")`` for each inner group (not
    the first or last) whose median ratio is the lowest or highest of all the
    groups. Used by both ``notes`` and the box plot, so they always agree.
    """
    if len(groups) <= 2:
        return []  # no inner groups
    medians = groups["Median_Ratio"].to_numpy()
    first_and_last = medians[[0, -1]]
    lowest = 1 + int(np.argmin(medians[1:-1]))   # the first of any tie, like idxmin
    highest = 1 + int(np.argmax(medians[1:-1]))

    extremes = []
    if medians[lowest] < first_and_last.min():
        extremes.append((lowest, "lowest"))
    if medians[highest] > first_and_last.max():
        extremes.append((highest, "highest"))
    return extremes


def _vei_notes(groups: pd.DataFrame) -> Tuple[str, ...]:
    """The notes of Appendix E.4 (see ``_vei_inner_extremes``), in plain English."""
    notes = []
    for position, word in _vei_inner_extremes(groups):
        row = groups.iloc[position]
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
    kruskal_wallis: bool = False,
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
    kruskal_wallis : bool, default False
        If True, also run the Kruskal-Wallis test across all the value groups
        (Appendix E.4) and store it in ``result.kruskal_wallis``; see
        ``Kruskal_Wallis``. It is information only and never changes the VEI's
        conclusion.
    show_plot : bool, default False
        If True, show the summary chart: the VEI and its confidence interval
        against the acceptable range, above a box plot of the sale ratios in
        each value group. ``result.plot(kind=...)`` draws either one on its
        own (see ``VEIResult.plot``).

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
    Kruskal_Wallis : the test behind ``kruskal_wallis=True``, for any groups.

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
    ratio, which the Standard says to review. With ``kruskal_wallis=True``,
    the Kruskal-Wallis test checks whether the level of valuation differs
    across all the groups, not only the first and last. The Standard adds a
    caution, repeated in the report: keep the statistical considerations of
    multiple comparisons in mind, and note that samples with higher
    dispersion can mask differences, especially when they are small.

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
    >>> result.plot()                         # the test and the value groups, charted
    >>> sales = sales.join(result.sales["Group"])   # each sale's value group
    >>> rem.VEI(sales["Sale_Price"], sales["Appraised_Value"],
    ...         kruskal_wallis=True).kruskal_wallis.p_value
    """
    ranking_method = _check_vei_options(confidence_level, ranking_method)

    # Check the data (see "Data checks" in the module documentation). Sales
    # with an appraised value of 0 or less are not kept here, as they are for
    # the other statistics: the VEI cannot use them, so they are left out.
    sale_price, appraised_value, row_labels, _ = _sales_pairs(
        Sale_Amount, Appr_Value, return_rows=True, report_unusual_appraised=False
    )
    sale_price, appraised_value, row_labels = _vei_leave_out_ratios_of_zero_or_less(
        sale_price, appraised_value, row_labels
    )

    result = _vei_calculation(sale_price, appraised_value, row_labels, confidence_level, ranking_method)
    if kruskal_wallis and result.scenario is not None:
        # Appendix E.4: does the level of valuation differ across all the value groups?
        test = Kruskal_Wallis(result.sales["Sale_Amount"], result.sales["Appr_Value"], result.sales["Group"])
        result = replace(result, kruskal_wallis=test)
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
