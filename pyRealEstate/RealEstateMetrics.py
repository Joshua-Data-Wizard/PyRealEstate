"""
Ratio-study statistics for mass appraisal: the median, mean and weighted
mean sale ratios and the COD, each with its confidence interval; whether the
COD meets the Standard for a property type; the MAPE, MdAPE, coefficient of
concentration and PE10 / PE20; the PRD, PRB, VEI; a test of normality; and
the Kruskal-Wallis and Mann-Whitney tests.

Argument convention used by every function in this module::

    y  (VEI: Sale_Amount) : sale prices
    x  (VEI: Appr_Value)  : appraised / model-predicted values
    sale ratio            = appraised value / sale price
    groups (Kruskal_Wallis and Mann_Whitney only) : the group of each sale

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
* ``Kruskal_Wallis`` and ``Mann_Whitney`` also leave out sales without a
  group label, and report them.
* A confidence interval that can't be calculated (for example, with too few
  sales) is NaN, and a ``DataQualityWarning`` says why.
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
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple, Union

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

# Deprecated names keep working until this version, with a FutureWarning. Python always shows a
# FutureWarning; a DeprecationWarning is hidden when the call comes from inside the user's own module.
_REMOVED_IN = "0.4.0"


def _warn_deprecated(old_name: str, new_name: str, advice: str) -> None:
    """
    Warn, from inside the deprecated function ``old_name``, that it will be
    removed: '<old_name> is deprecated and will be removed in pyRealEstate
    0.4.0. Use <new_name>, <advice>.' The warning points at the user's line.
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in pyRealEstate {_REMOVED_IN}. Use {new_name}, {advice}.",
        FutureWarning,
        stacklevel=3,  # past this helper and the deprecated function, to the line that called it
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


# ---------------------------------------------------------------------------
# Shared by the confidence intervals (Standard on Ratio Studies 2026,
# Appendix D)
# ---------------------------------------------------------------------------

def _two_tailed_critical_value(confidence_level: float, degrees_of_freedom: Optional[int] = None) -> float:
    """
    The two-tailed critical value for ``confidence_level``: Student's t with
    ``degrees_of_freedom`` (Table D.5.1), or the normal z when it is None
    (1.645 for 90%, 1.960 for 95%).
    """
    from scipy import stats  # imported here: slow to import (see top of module)

    upper_tail = (1 + confidence_level) / 2  # e.g. 0.95 for a 90% interval
    if degrees_of_freedom is None:
        return float(stats.norm.ppf(upper_tail))
    return float(stats.t.ppf(upper_tail, degrees_of_freedom))


def _no_interval(reason: str) -> Tuple[float, float]:
    """Warn that a confidence interval can't be calculated, and return it empty: (NaN, NaN)."""
    _warn_data_quality(f"{reason} The confidence interval is left empty (NaN).")
    return math.nan, math.nan


def Median_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Median sale ratio: the level of valuation the Standard prefers.

    The middle sale ratio when the ratios are put in order; with an even
    number of sales, the average of the two middle ratios. Half the sales
    have a ratio at or below it and half at or above it, so a few extreme
    ratios barely move it. The IAAO Standard on Ratio Studies (2026) calls
    it "the generally preferred measure of central tendency for evaluating
    overall valuation level" (Section 7.1).

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The median sale ratio. For example, 0.95 means the typical sale is
        appraised at 95% of its sale price.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    See Also
    --------
    Median_Ratio_Lower, Median_Ratio_Upper : its confidence interval.
    Mean_Ratio, Weighted_Mean_Ratio : the Standard's other measures of
        level (Section 7).
    COD : how far the ratios typically are from this median.

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Section 7.1)::

        ratio  = appraised value / sale price                    (x / y)
        median = the middle ratio, in order of size; for an even number
                 of sales, the average of the two middle ratios

    The Standard's example: the ratios 0.82, 0.92, 0.99, 1.05 and 1.15 have
    a median of 0.99. For its 54 example sales (Table 10.1), the median is
    0.8678 (Table 10.2).

    The Standard's level of valuation should be between 0.90 and 1.10 (the
    footnote of Table 11.1). The median of a sample is a point estimate: the
    Standard asks for a confidence interval before concluding that a level
    is out of compliance (Sections 7 and 7.5.1, Appendix D); see
    ``Median_Ratio_Lower``.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.Median_Ratio(sales["Sale_Price"], sales["Appraised_Value"])
    >>> rem.Median_Ratio([100_000] * 5, [82_000, 92_000, 99_000, 105_000, 115_000])  # 0.99
    """
    y, x = _sales_pairs(y, x)
    return np.median(x / y)


_MEDIAN_SMALL_SAMPLE = 30  # Appendix D.4: the binomial method is for "small samples with up to 30 observations"


def _binomial_coverage(n: int, d: int) -> float:
    """
    How sure the range from the d-th lowest to the d-th highest of n ratios
    is to contain the median of all the properties: P(d <= B <= n - d), with
    B ~ Binomial(n, 1/2), which is 1 - 2 * P(B < d).
    """
    below = sum(math.comb(n, k) for k in range(d))  # the ways fewer than d of n ratios fall below the median
    return 1 - 2 * below / 2**n


def _median_rank_interval(sorted_ratios: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """Appendix D.2 (more than 30 sales): count j ratios down and up from the median."""
    n = len(sorted_ratios)
    z = _two_tailed_critical_value(confidence_level)

    # Step 2: j = z * sqrt(n) / 2, plus 0.5 when n is even. Step 3: round up.
    j = math.ceil(z * math.sqrt(n) / 2 + (0.5 if n % 2 == 0 else 0.0))

    # Steps 4-5: the j-th ratio below and the j-th above the median. With an even n the median lies
    # between the two middle ratios, which count as the first below and the first above it.
    lower_rank = (n + 1) // 2 - j if n % 2 == 1 else n // 2 + 1 - j
    upper_rank = n + 1 - lower_rank
    if lower_rank < 1:
        return _no_interval(
            f"There are too few sales ({n}) for a {_percent(confidence_level)} confidence interval of the "
            f"median ratio (Standard on Ratio Studies, Appendix D.2): it would need {j} ratios below the median."
        )
    return sorted_ratios[lower_rank - 1], sorted_ratios[upper_rank - 1]  # ranks start at 1


def _small_sample_median_interval(sorted_ratios: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """
    Appendix D.4 (30 or fewer sales): Clapp's (1989) binomial interpolation,
    the calculation behind the Standard's Tables D.4.1 (90%) and D.4.2 (95%),
    for any confidence level.
    """
    n = len(sorted_ratios)
    if _binomial_coverage(n, 1) < confidence_level:
        minimum = n + 1
        while _binomial_coverage(minimum, 1) < confidence_level:
            minimum += 1
        return _no_interval(
            f"A {_percent(confidence_level)} confidence interval of the median ratio needs at least {minimum} "
            f"sales (Standard on Ratio Studies, Appendix D.4), but only {n} usable sales were given: with "
            f"{n}, even the range from the lowest to the highest ratio is less than {_percent(confidence_level)} "
            "sure to contain the median."
        )

    # The narrowest range R_d .. R_(n+1-d) that is still at least as sure as asked...
    d = max(d for d in range(1, n // 2 + 1) if _binomial_coverage(n, d) >= confidence_level)
    # ...then each limit moves toward the next ratio in, in proportion to the certainty to spare. (For an
    # even n, the step in from the two middle ratios crosses over: the coverage formula turns negative and
    # both limits move toward the median instead. That only happens below 50% confidence.)
    spare = _binomial_coverage(n, d) - confidence_level
    weight = spare / (_binomial_coverage(n, d) - _binomial_coverage(n, d + 1))
    lower = (1 - weight) * sorted_ratios[d - 1] + weight * sorted_ratios[d]          # R_d and R_(d+1)
    upper = (1 - weight) * sorted_ratios[n - d] + weight * sorted_ratios[n - d - 1]  # R_(n+1-d) and R_(n-d)
    return lower, upper


def _median_ratio_interval(y: Any, x: Any, confidence_level: float) -> Tuple[float, float]:
    """The median ratio's two-tailed confidence interval, (lower, upper): Appendix D.2 or D.4."""
    y, x = _sales_pairs(y, x)
    sorted_ratios = np.sort(x / y)
    if len(sorted_ratios) > _MEDIAN_SMALL_SAMPLE:
        return _median_rank_interval(sorted_ratios, confidence_level)
    return _small_sample_median_interval(sorted_ratios, confidence_level)


def Median_Ratio_Lower(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Lower limit of the median ratio's confidence interval (90% by default).

    The median of a sample of sales estimates the median ratio of all the
    properties the sales represent. Its confidence interval is the range
    that contains that true median with the confidence you choose. The IAAO
    Standard on Ratio Studies (2026) calls a point estimate "only an
    indication, not proof" of the level of valuation, and uses confidence
    intervals to decide compliance (Sections 11.1 and 11.1.2).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The lower confidence limit of the median ratio. NaN when there are
        too few sales for an interval at this confidence level: a 90%
        interval needs at least 5 sales, a 95% interval 6 (see Notes).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Median_Ratio : the median ratio.
    Median_Ratio_Upper : the upper limit.

    Notes
    -----
    The interval is read from the sale ratios themselves, sorted from the
    lowest, R_1, to the highest, R_n, so it does not assume that the ratios
    follow a normal distribution. The Standard gives one method for more
    than 30 sales and another for 30 or fewer (Appendix D).

    More than 30 sales (Appendix D.2)::

        j = z * sqrt(n) / 2             when n is odd
        j = z * sqrt(n) / 2 + 0.5       when n is even
        j is rounded up to a whole number
        lower limit = the j-th ratio below the median
        upper limit = the j-th ratio above the median

    z is the two-tailed normal critical value (1.645 for 90%, 1.960 for
    95%). With an even number of sales the median lies between the two
    middle ratios, which count as the first below and the first above it.
    Rounding j up makes the interval slightly cautious.

    30 or fewer sales (Appendix D.4, the binomial method of Clapp 1989)::

        coverage(d) = P(d <= B <= n - d),  B ~ Binomial(n, 1/2)
        d           = the largest d with coverage(d) >= confidence_level
        w           = (coverage(d) - confidence_level)
                      / (coverage(d) - coverage(d + 1))
        lower limit = (1 - w) * R_d       + w * R_(d+1)
        upper limit = (1 - w) * R_(n+1-d) + w * R_(n-d)

    coverage(d) is the probability that the range from the d-th lowest to
    the d-th highest ratio contains the true median, and w moves each limit
    toward the next ratio in, in proportion to the coverage to spare. This
    is how the Standard's Tables D.4.1 (90%) and D.4.2 (95%) were made: it
    gives every weight in Table D.4.1, and every weight in Table D.4.2
    except two misprints (n = 28: .2729 should be .2779; n = 29: the upper
    .3062 should be .3063). Calculating it also works for other confidence
    levels. Even the range from the lowest to the highest ratio contains
    the median with a probability of only 1 - 2 / 2**n, so a 90% interval
    needs at least 5 sales and a 95% interval at least 6.

    For the Standard's 54 example sales (Table 10.1) the 90% interval is
    (0.7867, 0.9840) (Table 10.2): j = 7, the 21st and the 34th ratios. For
    the 17 sales of its small-sample example (Table D.4.3) the 95% interval
    is (0.6956, 0.9329); the Standard prints (0.6952, 0.9326) because it
    used the ratios rounded to 3 decimals.

    The method changes between 30 and 31 sales, as the Standard specifies,
    so the interval can jump there.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.Median_Ratio_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    >>> lower_95 = rem.Median_Ratio_Lower(sales["Sale_Price"], sales["Appraised_Value"],
    ...                                   confidence_level=0.95)
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    lower, _ = _median_ratio_interval(y, x, confidence_level)
    return lower


def Median_Ratio_Upper(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Upper limit of the median ratio's confidence interval (90% by default).

    See ``Median_Ratio_Lower`` for what the interval means and how it is
    calculated (Standard on Ratio Studies 2026, Appendix D.2 and D.4).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The upper confidence limit of the median ratio. NaN when there are
        too few sales for an interval at this confidence level (5 for 90%,
        6 for 95%).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Median_Ratio : the median ratio.
    Median_Ratio_Lower : the lower limit, and the formulas.

    Notes
    -----
    For the Standard's 54 example sales (Table 10.1) the 90% interval is
    (0.7867, 0.9840) (Table 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.Median_Ratio_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    _, upper = _median_ratio_interval(y, x, confidence_level)
    return upper


def Mean_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Mean sale ratio: the simple average of the sale ratios.

    Every sale counts equally. Unlike the median, the mean is pulled by
    extreme ratios: one sale appraised at five times its price moves it far
    more than it moves the median. The IAAO Standard on Ratio Studies (2026)
    says the mean "can be relied upon only if outliers have been
    appropriately trimmed" (Section 7.4).

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The mean sale ratio.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    See Also
    --------
    Mean_Ratio_Lower, Mean_Ratio_Upper : its confidence interval.
    Median_Ratio, Weighted_Mean_Ratio : the Standard's other measures of
        level (Section 7).
    PRD : the mean ratio divided by the weighted mean ratio.

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Section 7.2)::

        ratio = appraised value / sale price                    (x / y)
        mean  = sum(ratios) / number of sales

    The Standard's example: the ratios 0.82, 0.92, 0.99, 1.05 and 1.15 have
    a mean of 0.986. For its 54 example sales (Table 10.1), the mean is
    0.8938 (Table 10.2). Sale ratios are usually skewed to the right, so the
    mean tends to be above the median (Section 7.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.Mean_Ratio(sales["Sale_Price"], sales["Appraised_Value"])
    >>> rem.Mean_Ratio([100_000] * 5, [82_000, 92_000, 99_000, 105_000, 115_000])  # 0.986
    """
    y, x = _sales_pairs(y, x)
    return np.mean(x / y)


def _mean_ratio_interval(y: Any, x: Any, confidence_level: float) -> Tuple[float, float]:
    """The mean ratio's two-tailed confidence interval, (lower, upper): Appendix D.1."""
    y, x = _sales_pairs(y, x)
    ratio = x / y
    n = len(ratio)
    if n < 2:
        return _no_interval(
            "The confidence interval of the mean ratio needs at least 2 sales (Standard on Ratio Studies, "
            "Appendix D.1), but only 1 usable sale was given."
        )
    t = _two_tailed_critical_value(confidence_level, degrees_of_freedom=n - 1)
    margin_of_error = t * np.std(ratio, ddof=1) / math.sqrt(n)  # t * the standard error of the mean
    mean = np.mean(ratio)
    return mean - margin_of_error, mean + margin_of_error


def Mean_Ratio_Lower(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Lower limit of the mean ratio's confidence interval (90% by default).

    The range that contains the mean ratio of all the properties the sales
    represent, with the confidence you choose (IAAO Standard on Ratio
    Studies 2026, Appendix D.1). See ``Median_Ratio_Lower`` for why the
    Standard asks for confidence intervals.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The lower confidence limit of the mean ratio. NaN with fewer than 2
        sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Mean_Ratio : the mean ratio.
    Mean_Ratio_Upper : the upper limit.
    Median_Ratio_Lower : the median's interval, which assumes no normal
        distribution.
    Normality_Test : are the ratios normally distributed?

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Appendix D.1)::

        mean ratio -/+ t * s / sqrt(n)

    where s is the standard deviation of the sale ratios (dividing by
    n - 1), n the number of sales, and t the two-tailed Student-t critical
    value with n - 1 degrees of freedom (Table D.5.1: 2.132 for 90% and 5
    sales). The Standard allows 1.645 (90%) or 1.96 (95%) as approximations
    above 30 sales, but its own report uses the exact t, as this function
    does: for its 54 example sales (Table 10.1) the 90% interval is
    (0.8350, 0.9526) (Table 10.2), where 1.645 would give (0.8360, 0.9516).

    The formula assumes a normal distribution (D.1). Sale ratios are often
    skewed, and the mean is pulled by extreme ratios (Section 7.4), so with
    few sales or outliers the median's interval is the safer guide.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.Mean_Ratio_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    lower, _ = _mean_ratio_interval(y, x, confidence_level)
    return lower


def Mean_Ratio_Upper(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Upper limit of the mean ratio's confidence interval (90% by default).

    See ``Mean_Ratio_Lower`` for the formula (Standard on Ratio Studies
    2026, Appendix D.1).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The upper confidence limit of the mean ratio. NaN with fewer than 2
        sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Mean_Ratio : the mean ratio.
    Mean_Ratio_Lower : the lower limit, and the formula.

    Notes
    -----
    For the Standard's 54 example sales (Table 10.1) the 90% interval is
    (0.8350, 0.9526) (Table 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.Mean_Ratio_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    _, upper = _mean_ratio_interval(y, x, confidence_level)
    return upper


def Weighted_Mean_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Weighted mean ratio: the total of the appraised values divided by the
    total of the sale prices.

    Because it works with totals, higher-priced sales carry more weight than
    in the simple average of the sale ratios (``Mean_Ratio``). The IAAO
    Standard on Ratio Studies (2026) describes it as "the value-weighted
    average of the ratios in which the weights are proportional to the sale
    prices", also called the aggregate ratio (Section 7.3).

    .. versionadded:: 0.3.0
       Replaces ``weighted_Mean_Sale_Ratio``, which gives the same result
       and will be removed in pyRealEstate 0.4.0. The new name matches
       ``Median_Ratio``, ``Mean_Ratio`` and ``Weighted_Mean_Lower`` /
       ``Weighted_Mean_Upper``.

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

    See Also
    --------
    Weighted_Mean_Lower, Weighted_Mean_Upper : its confidence interval.
    Median_Ratio, Mean_Ratio : the Standard's other measures of level
        (Section 7).

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Section 7.3)::

        weighted mean = sum(appraised values) / sum(sale prices)
                      = mean(x) / mean(y)

    The Standard's example (Section 7.3): values of 2,245,000 and sale
    prices of 2,333,000 give 0.962. For its 54 example sales (Table 10.1),
    the weighted mean is 0.9003 (Table 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.Weighted_Mean_Ratio(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    return np.mean(x) / np.mean(y)


def weighted_Mean_Sale_Ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Weighted mean sale ratio, under its old name: use ``Weighted_Mean_Ratio``.

    .. deprecated:: 0.3.0
       ``weighted_Mean_Sale_Ratio`` will be removed in pyRealEstate 0.4.0.
       Use ``Weighted_Mean_Ratio``, which gives the same result; the new
       name matches ``Median_Ratio`` and ``Mean_Ratio``.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The weighted mean sale ratio (Standard on Ratio Studies 2026,
        Section 7.3), exactly as ``Weighted_Mean_Ratio`` returns it.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    FutureWarning
        Always, to say that the name is deprecated. Python shows it once for
        each line of your code that calls it.
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.Weighted_Mean_Ratio(sales["Sale_Price"], sales["Appraised_Value"])  # instead of this function
    """
    _warn_deprecated("weighted_Mean_Sale_Ratio", "Weighted_Mean_Ratio", "which gives the same result")
    return Weighted_Mean_Ratio(y, x)


def _weighted_mean_interval(y: Any, x: Any, confidence_level: float) -> Tuple[float, float]:
    """The weighted mean ratio's two-tailed confidence interval, (lower, upper): Appendix D.3."""
    y, x = _sales_pairs(y, x)
    n = len(y)
    if n < 2:
        return _no_interval(
            "The confidence interval of the weighted mean ratio needs at least 2 sales (Standard on Ratio "
            "Studies, Appendix D.3), but only 1 usable sale was given."
        )
    weighted_mean = np.mean(x) / np.mean(y)  # exactly as Weighted_Mean_Ratio
    sale_price, appraised_value = y.astype(float), x.astype(float)

    # Steps 1-4 and 10-12 of D.3: sqrt(sum(A^2) - 2w * sum(A*S) + w^2 * sum(S^2)). That is the same
    # number as sqrt(sum((A - w*S)^2)), which is used here: it can't lose precision by subtracting
    # large totals.
    spread = math.sqrt(np.sum((appraised_value - weighted_mean * sale_price) ** 2))

    # Steps 8, 13 and 14: t times the spread, divided by the average sale price * sqrt(n(n - 1))
    t = _two_tailed_critical_value(confidence_level, degrees_of_freedom=n - 1)
    margin_of_error = t * spread / (np.mean(sale_price) * math.sqrt(n * (n - 1)))
    return weighted_mean - margin_of_error, weighted_mean + margin_of_error


def Weighted_Mean_Lower(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Lower limit of the weighted mean ratio's confidence interval (90% by default).

    The range that contains the weighted mean ratio (total appraised value
    divided by total sale price) of all the properties the sales represent,
    with the confidence you choose (IAAO Standard on Ratio Studies 2026,
    Appendix D.3). See ``Median_Ratio_Lower`` for why the Standard asks for
    confidence intervals.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The lower confidence limit of the weighted mean ratio. NaN with
        fewer than 2 sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Weighted_Mean_Ratio : the weighted mean ratio.
    Weighted_Mean_Upper : the upper limit.
    Normality_Test : are the ratios normally distributed?

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Appendix D.3), with A the
    appraised values, S the sale prices, n the number of sales and
    w = sum(A) / sum(S) the weighted mean::

        w -/+ t * sqrt(sum(A**2) - 2 * w * sum(A * S) + w**2 * sum(S**2))
                / (mean(S) * sqrt(n * (n - 1)))

    where t is the two-tailed Student-t critical value with n - 1 degrees
    of freedom. The square root is the same number as
    sqrt(sum((A - w * S)**2)), which is how it is calculated here: that
    form can't lose precision by subtracting large totals. For the
    Standard's 54 example sales (Table 10.1) the 90% interval is (0.8359,
    0.9647) (Table 10.2).

    Caution: this is a large-sample formula. In simulations with skewed
    (lognormal) ratios, its 90% interval contained the true weighted mean
    only 85-88% of the time with 16 or fewer sales, and 88-90% from about
    50 sales, so treat it as too narrow for small samples.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.Weighted_Mean_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    lower, _ = _weighted_mean_interval(y, x, confidence_level)
    return lower


def Weighted_Mean_Upper(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Upper limit of the weighted mean ratio's confidence interval (90% by default).

    See ``Weighted_Mean_Lower`` for the formula and a caution about small
    samples (Standard on Ratio Studies 2026, Appendix D.3).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The upper confidence limit of the weighted mean ratio. NaN with
        fewer than 2 sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if there are too few sales
        for the interval (it is then NaN).

    See Also
    --------
    Weighted_Mean_Ratio : the weighted mean ratio.
    Weighted_Mean_Lower : the lower limit, and the formula.

    Notes
    -----
    For the Standard's 54 example sales (Table 10.1) the 90% interval is
    (0.8359, 0.9647) (Table 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.Weighted_Mean_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    _, upper = _weighted_mean_interval(y, x, confidence_level)
    return upper


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

    See Also
    --------
    COD_Lower, COD_Upper : its confidence interval.
    COD_Conclusion, COD_Cutoff : whether it meets the Standard for a
        property type (Section 11.2, Appendix D.8).

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Section 8.1.1)::

        ratio = appraised value / sale price                    (x / y)
        COD   = 100 * mean(|ratio - median ratio|) / median ratio

    For the Standard's 54 example sales (Table 10.1), the COD is 24.65%
    (Table 10.2).

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


_COD_INTERVAL_MINIMUM_SALES = 5  # with 4 sales, step 8 of Appendix D.6 gives the rank round(0.5) = 0
_COD_T_BELOW = 30  # Appendix D.6: "Refer to a t-table for the appropriate t-value for samples less than 30"


def _cod_log_impossible(rank: int, n: int, ratio: float) -> Tuple[float, float]:
    """The COD's interval needs the logarithm of a ratio that is 0 or less: warn, and return (NaN, NaN)."""
    return _no_interval(
        f"The COD's confidence interval takes the logarithm of the sale ratio ranked {rank} of {n}, which "
        f"is {ratio:.4g}; logarithms only exist above 0 (such ratios come from appraised values of 0 or less)."
    )


def _cod_rank_a(n: int) -> int:
    """
    Appendix D.6, step 8: the rank a of the ratio below the median whose log
    measures the median's uncertainty. round() sends an exact half to the
    even number (4.5 -> 4), as the method's authors do.
    """
    return round((n + 1) / 2 - math.sqrt(n))


def _cod_log_variances(ratios: np.ndarray, median: float, tau: float) -> Tuple[float, float, float]:
    """
    Appendix D.6, steps 5-14 (Bonett and Seier 2006), for sorted ratios with
    median ``median`` and mean absolute deviation ``tau`` above 0, whose ratio
    ranked a (step 8) is above 0 (the caller checks both): the variance of the
    median on the log scale, the variance of tau on the log scale, and their
    covariance. Together they give the variance of log(COD), since
    log(COD) = log(tau) - log(median):

        var(log COD) = var_log_median + var_log_tau - 2 x covariance

    Shared by the COD's confidence interval and ``Sales_Chasing``'s
    split-sample test.
    """
    n = len(ratios)
    mean = float(np.mean(ratios))
    variance = float(np.var(ratios, ddof=1))
    delta = (mean - median) / tau  # step 5: the mean-median standardized difference
    gamma = variance / tau**2      # step 6: the relative dispersion ratio

    # Steps 8-11: the uncertainty of the median on the log scale, from the ratios ranked a and b
    rank_a = _cod_rank_a(n)
    rank_b = n - rank_a + 1
    var_log_median = ((math.log(ratios[rank_a - 1]) - math.log(ratios[rank_b - 1])) / 4) ** 2

    # Steps 12-13: the uncertainty of tau on the log scale, from the moments of the ratios
    var_log_tau = (gamma + delta**2 - 1) / n

    # Step 14: their covariance. (The Standard prints "(2 + covltle)" in step 15; its worked
    # example, Table D.7.1, needs 2 x covltle.)
    covariance = delta * math.sqrt(var_log_median) / math.sqrt(n)
    return var_log_median, var_log_tau, covariance


def _cod_interval(y: Any, x: Any, confidence_level: float) -> Tuple[float, float]:
    """
    The COD's two-tailed confidence interval, (lower, upper), in percent:
    Appendix D.6 (Bonett and Seier 2006). The steps are numbered as in the
    Standard; ``COD_Lower`` lists the formulas.
    """
    y, x = _sales_pairs(y, x)
    ratios = np.sort(x / y)
    n = len(ratios)
    if n < _COD_INTERVAL_MINIMUM_SALES:
        return _no_interval(
            f"The confidence interval of the COD needs at least {_COD_INTERVAL_MINIMUM_SALES} sales (Standard "
            f"on Ratio Studies, Appendix D.6), but only {n} usable sales were given."
        )

    # Steps 1-7: the median of the ratios and their mean absolute deviation from it (tau). Plain
    # Python floats from here on, so round() below gives whole numbers.
    median = float(np.median(ratios))
    tau = float(np.mean(np.abs(ratios - median)))
    if tau == 0:
        return _no_interval(
            f"Every sale ratio is the same ({ratios[0]:.4f}), so the COD is 0 and its confidence interval "
            "can't be calculated (Appendix D.6 divides by the spread of the ratios). Identical ratios are "
            "unusual in real data; please check them (for example, for sales chasing)."
        )
    rank_a = _cod_rank_a(n)
    if ratios[rank_a - 1] <= 0:
        return _cod_log_impossible(rank_a, n, ratios[rank_a - 1])

    # Steps 5-14: the uncertainty of the median and of tau on the log scale, and their covariance
    var_log_median, var_log_tau, covariance = _cod_log_variances(ratios, median, tau)
    se_log_median, se_log_tau = math.sqrt(var_log_median), math.sqrt(var_log_tau)

    # Step 15: the factor k that combines them
    k = math.sqrt(var_log_median + var_log_tau - 2 * covariance) / (se_log_median + se_log_tau)

    # The critical value: z, or Student's t with fewer than 30 sales (the Standard's note to D.6)
    if n < _COD_T_BELOW:
        critical_value = _two_tailed_critical_value(confidence_level, degrees_of_freedom=n - 1)
    else:
        critical_value = _two_tailed_critical_value(confidence_level)

    # Steps 16-17: the ratios ranked low and high bound the median
    rank_low = round((n + 1) / 2 - k * critical_value * math.sqrt(n / 4))
    rank_high = n - rank_low + 1
    if rank_low < 1:
        return _no_interval(
            f"There are too few sales ({n}) for a {_percent(confidence_level)} confidence interval of the COD "
            "(Standard on Ratio Studies, Appendix D.6: the rank of step 16 falls below 1)."
        )
    if ratios[rank_low - 1] <= 0:
        return _cod_log_impossible(rank_low, n, ratios[rank_low - 1])

    # Step 18: the limits of tau on the log scale (tau scaled by n / (n - 1))
    log_tau = math.log(n / (n - 1) * tau)
    lower_log_tau = log_tau - k * critical_value * se_log_tau
    upper_log_tau = log_tau + k * critical_value * se_log_tau

    # Step 19: the COD is tau / median, so its lower limit pairs tau's lower limit with the median's
    # upper one, and its upper limit tau's upper limit with the median's lower one. (The Standard
    # prints Ratio_RankHigh for both; the published upper limit needs Ratio_RankLow.)
    lower = math.exp(lower_log_tau - math.log(ratios[rank_high - 1]))
    upper = math.exp(upper_log_tau - math.log(ratios[rank_low - 1]))
    return 100 * lower, 100 * upper


def COD_Lower(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Lower limit of the COD's confidence interval, in percent (90% by default).

    The COD of a sample of sales estimates the COD of all the properties
    the sales represent. Its confidence interval is the range that contains
    that true COD with the confidence you choose. The IAAO Standard on
    Ratio Studies (2026) says a COD "is a point estimate and an indicator,
    but not proof of non-compliance", and names this interval as proof
    (Section 11.2, Appendix D.6).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The lower confidence limit of the COD, in percent like ``COD``. NaN
        when the interval can't be calculated (see Notes).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if the interval can't be
        calculated (it is then NaN).

    See Also
    --------
    COD : the COD.
    COD_Upper : the upper limit.

    Notes
    -----
    The method of Bonett and Seier (2006), as Appendix D.6 of the Standard
    lays it out. It works on the log scale and combines the uncertainty of
    the median (from the ratios around it) with that of the mean absolute
    deviation (from the moments of the ratios). With the n sale ratios
    sorted from R_1 (lowest) to R_n, and the Standard's step numbers::

        median, mean, s**2 of the ratios (s**2 divides by n - 1)     1-2
        tau    = mean(|ratio - median|)                              3-4
        delta  = (mean - median) / tau                               5
        gamma  = s**2 / tau**2                                       6
        a      = round((n + 1) / 2 - sqrt(n)),   b = n - a + 1       8
        var_m  = ((ln R_a - ln R_b) / 4)**2,     se_m = sqrt(var_m)  9-11
        var_t  = (gamma + delta**2 - 1) / n,     se_t = sqrt(var_t)  12-13
        cov    = delta * se_m / sqrt(n)                              14
        k      = sqrt(var_m + var_t - 2 * cov) / (se_m + se_t)       15
        low    = round((n + 1) / 2 - k * c * sqrt(n / 4))            16
        high   = n - low + 1                                         16
        L1     = ln(n / (n - 1) * tau) - k * c * se_t                18
        U1     = ln(n / (n - 1) * tau) + k * c * se_t                18
        lower  = 100 * exp(L1 - ln R_high)                           19
        upper  = 100 * exp(U1 - ln R_low)                            19

    c is the two-tailed critical value: the normal z (1.645 for 90%), or,
    with fewer than 30 sales, Student's t with n - 1 degrees of freedom, as
    the Standard's note to D.6 says. (Bonett and Seier use z for every
    sample size; t makes small-sample intervals wider, so more cautious.)
    round() sends an exact half to the even number (4.5 becomes 4), as the
    method's authors do; it only matters with 16, 36, 64, 100, ... sales.
    The interval is built on the log scale, so it is not centered on the
    COD.

    Two formulas above differ from the printed text of D.6, which has two
    misprints: step 15 prints "(2 + covltle)" where the formula needs
    2 * cov, and step 19 uses the high-rank ratio for both limits. The
    formulas above reproduce every value of the Standard's worked example,
    Table D.7.1; the printed ones do not (the printed step 15 is the square
    root of a negative number there). For the Standard's 54 example sales
    (Table 10.1), the 90% interval is 21.878% to 28.618% (Tables D.7.1 and
    10.2).

    The interval needs at least 5 sales (with 4, step 8 gives rank 0), and
    it is NaN, with a warning, when every ratio is the same (the COD is 0),
    when a ratio whose logarithm it needs is 0 or less (from appraised
    values of 0 or less), or when step 16's rank falls below 1, which
    happens when k * c exceeds sqrt(n): with a large t and few sales. A 90%
    interval is always possible from 5 sales, a 95% interval from 7.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.COD_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    lower, _ = _cod_interval(y, x, confidence_level)
    return lower


def COD_Upper(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> float:
    """
    Upper limit of the COD's confidence interval, in percent (90% by default).

    See ``COD_Lower`` for what the interval means and the formulas
    (Standard on Ratio Studies 2026, Appendix D.6).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1. The
        Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    float
        The upper confidence limit of the COD, in percent like ``COD``. NaN
        when the interval can't be calculated (see ``COD_Lower``).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if the interval can't be
        calculated (it is then NaN).

    See Also
    --------
    COD : the COD.
    COD_Lower : the lower limit, and the formulas.

    Notes
    -----
    For the Standard's 54 example sales (Table 10.1), the 90% interval is
    21.878% to 28.618% (Tables D.7.1 and 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.COD_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    _, upper = _cod_interval(y, x, confidence_level)
    return upper


# ---------------------------------------------------------------------------
# Does the COD meet the Standard? (Section 11.2; Appendix D.6 and D.8)
# ---------------------------------------------------------------------------

_COD_TEST_METHODS = ("interval", "tolerance")


def _required_cod_standard(property_type: Any, function_name: str) -> "_CodStandard":
    """The COD range for ``property_type``. Unlike the summary's, it is required: a test needs a standard."""
    standard = _cod_standard(property_type)
    if standard is None:
        raise ValueError(
            f"{function_name} needs a COD standard to compare with: name the property type and market (for "
            "example 'residential'; Ratio_Study_Report lists every name) or give your own (low, high) "
            f"range, not {property_type!r}."
        )
    return standard


def _check_one_sided_level(confidence_level: float) -> None:
    """Raise a ValueError unless ``confidence_level`` is a one-sided level, above 0.5 and below 1."""
    _check_fraction(confidence_level, "confidence_level", "for example 0.95 for 95%")
    if confidence_level <= 0.5:
        raise ValueError(
            "confidence_level must be above 0.5: it is one-sided (for example 0.95 for 95%), "
            f"not {confidence_level!r}."
        )


def _check_cod_test_method(method: Any) -> str:
    """'interval' or 'tolerance', in any capitalization; anything else raises a ValueError."""
    if isinstance(method, str) and method.strip().lower() in _COD_TEST_METHODS:
        return method.strip().lower()
    raise ValueError(
        "method must be 'interval' (the COD's confidence interval, Appendix D.6) or 'tolerance' "
        f"(the tolerance factor, Appendix D.8), not {method!r}."
    )


def _cod_verdict(compared: float, limit: float) -> Optional[str]:
    """
    The COD tests' decision rule, shared by ``COD_Conclusion`` and the
    summary: 'FAIL' when ``compared`` (the COD's lower confidence limit, or
    the COD) is above ``limit`` (the maximum, or the cutoff), else 'PASS';
    None when either is NaN (too few sales for the test).
    """
    if math.isnan(compared) or math.isnan(limit):
        return None
    # Rounded, so floating-point noise can't push a value exactly at the limit past it
    if round(compared, 10) > round(limit, 10):
        return 'FAIL'
    return 'PASS'


def _cod_tolerance_factor(n: int, confidence_level: float) -> float:
    """
    Appendix D.8: TF = sqrt(chi-square / (n - 1)), with chi-square the value a
    chi-square distribution with n - 1 degrees of freedom exceeds with
    probability 1 - confidence_level (Excel: CHISQ.INV.RT(0.05, n - 1) for 95%).
    """
    from scipy import stats  # imported here: slow to import (see top of module)

    return math.sqrt(stats.chi2.ppf(confidence_level, n - 1) / (n - 1))


def COD_Cutoff(
    y: np.ndarray,
    x: np.ndarray,
    property_type: Union[str, Tuple[float, float]],
    confidence_level: float = 0.95,
) -> float:
    """
    The highest COD the sales can have before the Standard's tolerance test
    rejects the property type's standard (Appendix D.8).

    A COD a little above the maximum for a property type can still come from
    properties that meet the standard: a sample's COD varies from sample to
    sample. The tolerance test of the IAAO Standard on Ratio Studies (2026)
    allows for this by multiplying the maximum COD by a tolerance factor that
    shrinks as the number of sales grows. A COD above this cutoff means the
    standard is rejected; at or below it, it is not
    (``COD_Conclusion(..., method="tolerance")``).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    property_type : str or (float, float)
        The COD standard: a property type and market of the Standard's Tables
        11.1 and 11.2 (for example ``"residential"``; ``Ratio_Study_Report``
        lists every name with its range), or your own ``(low, high)`` range.
        The high end of the range is the maximum COD.
    confidence_level : float, default 0.95
        One-sided confidence level, above 0.5 and below 1. The Standard's
        Table D.8.1 uses 95%.

    Returns
    -------
    float
        The cutoff COD, in percent like ``COD``. NaN with fewer than 2 sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``property_type`` is "none", None, an unknown name, or not a valid
        range; or if ``confidence_level`` is not above 0.5 and below 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or with fewer than 2 sales.

    See Also
    --------
    COD_Conclusion : the test itself, by this cutoff or by the interval.
    COD_Lower, COD_Upper : the COD's confidence interval.

    Notes
    -----
    Formula (Standard on Ratio Studies 2026, Appendix D.8)::

        TF         = sqrt(chi-square / (n - 1))
        cutoff COD = TF * maximum COD

    where n is the number of sales and chi-square is the value a chi-square
    distribution with n - 1 degrees of freedom exceeds with probability
    1 - confidence_level (in Excel, ``CHISQ.INV.RT(0.05, n - 1)`` for 95%).
    For example, a maximum COD of 20 and 30 sales give TF = 1.2114 and a
    cutoff of 24.23, as in Table D.8.1 (the text's worked example multiplies
    the rounded 1.21 and prints 24.20). For the Standard's 54 example sales
    as residential property (maximum 15), the cutoff is 17.36.

    The test assumes approximately normally distributed ratios (D.8). In
    simulations of samples whose true COD was exactly the maximum, it
    rejected 4-8% of them (5% is intended), and 12-15% when 5% of the ratios
    were untrimmed outliers: trim outliers first (Appendix C), or use the
    interval test, ``COD_Conclusion``'s default, which held about 5% in every
    case.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.COD_Cutoff(sales["Sale_Price"], sales["Appraised_Value"], "residential")
    >>> rem.COD_Cutoff(sales["Sale_Price"], sales["Appraised_Value"], (5.0, 12.0))
    """
    _check_one_sided_level(confidence_level)
    standard = _required_cod_standard(property_type, "COD_Cutoff")
    y, x = _sales_pairs(y, x)
    n = len(y)
    if n < 2:
        _warn_data_quality(
            "The tolerance test of the COD needs at least 2 sales (Standard on Ratio Studies, Appendix D.8: "
            "its chi-square value has n - 1 degrees of freedom), but only 1 usable sale was given. The cutoff "
            "is left empty (NaN)."
        )
        return math.nan
    return _cod_tolerance_factor(n, confidence_level) * standard.high


def COD_Conclusion(
    y: np.ndarray,
    x: np.ndarray,
    property_type: Union[str, Tuple[float, float]],
    confidence_level: float = 0.95,
    method: str = "interval",
) -> Optional[str]:
    """
    Does the COD meet the Standard for a property type? ``'PASS'`` or ``'FAIL'``.

    ``'FAIL'`` only when the sales show, with the confidence you choose, that
    the COD of all the properties they represent is above the maximum for the
    property type. A sample's COD alone "is a point estimate and an
    indicator, but not proof of non-compliance" (IAAO Standard on Ratio
    Studies 2026, Section 11.2). The Standard gives two tests; ``method``
    chooses one.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    property_type : str or (float, float)
        The COD standard: a property type and market of the Standard's Tables
        11.1 and 11.2 (for example ``"residential"``; ``Ratio_Study_Report``
        lists every name with its range), or your own ``(low, high)`` range.
        The high end of the range is the maximum COD.
    confidence_level : float, default 0.95
        One-sided confidence level, above 0.5 and below 1. 95% one-sided is
        the same test as 90% two-tailed (Section 11.1.2).
    method : {"interval", "tolerance"}, default "interval"
        ``"interval"``: 'FAIL' when the COD's confidence interval lies
        entirely above the maximum (Section 11.2, Appendix D.6).
        ``"tolerance"``: 'FAIL' when the COD is above ``COD_Cutoff``
        (Appendix D.8). Any capitalization.

    Returns
    -------
    str or None
        ``'PASS'`` or ``'FAIL'``. ``None`` when the test can't be done: the
        interval test needs the COD's confidence interval (at least 5 sales;
        see ``COD_Lower``), the tolerance test at least 2 sales.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``property_type`` is "none", None, an unknown name, or not a valid
        range; if ``confidence_level`` is not above 0.5 and below 1; or if
        ``method`` is not "interval" or "tolerance".

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation; each problem is reported once),
        if the test can't be done, or if the COD is below the range for the
        property type (see Notes).

    See Also
    --------
    COD, COD_Lower, COD_Upper : the COD and its confidence interval.
    COD_Cutoff : the tolerance test's cutoff.

    Notes
    -----
    The decision rule, with the maximum COD the high end of the property
    type's range::

        method="interval":   'FAIL' if the lower limit of the COD's two-tailed
                             interval at 2 * confidence_level - 1 is above the
                             maximum (for 95%: COD_Lower(..., 0.90))
        method="tolerance":  'FAIL' if the COD is above COD_Cutoff(...)
                             (TF * maximum COD)
        otherwise            'PASS'

    A COD or limit exactly at the maximum or cutoff passes ("If Study COD <=
    Cutoff COD = Standard not rejected", D.8); values are compared after
    rounding to 10 decimal places. 'PASS' means the sales do not show that
    the COD is above the standard; it is not proof that it is within it.

    Why the interval test is the default: in simulations of samples whose
    true COD was exactly the maximum, it rejected about 5% of them, as
    intended, whatever the shape of the ratios; the tolerance test rejected
    4-8%, and 12-15% when 5% of the ratios were untrimmed outliers. The
    interval test also matches the 90% interval ``Ratio_Study_Report``
    shows, and it is the rule of ``PRB_Conclusion`` and ``VEI_Conclusion``.
    The tolerance test is the Standard's simple spreadsheet method: choose it
    to match a calculation done that way.

    A COD below the range's minimum is not a failure, but the Standard says
    it "should be treated as a diagnostic warning" (Section 11.2.5): it can
    indicate sales chasing or non-representative sales, so the function
    warns. (D.8 mentions a similar test for very low CODs but gives no
    formula, so none is done.)

    For the Standard's 54 example sales (Table 10.1) as residential property
    (5.0 to 15.0), both tests give 'FAIL': the COD's 90% interval is 21.878
    to 28.618, entirely above 15, and the COD, 24.65, is above the cutoff,
    17.36.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.COD_Conclusion(sales["Sale_Price"], sales["Appraised_Value"], "residential")
    >>> rem.COD_Conclusion(sales["Sale_Price"], sales["Appraised_Value"], "vacant land",
    ...                    method="tolerance")
    """
    _check_one_sided_level(confidence_level)
    standard = _required_cod_standard(property_type, "COD_Conclusion")
    method = _check_cod_test_method(method)

    # The COD and its interval (or cutoff) each check the data: collect their
    # data-quality warnings and report each problem once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataQualityWarning)
        cod = COD(y, x)
        if method == "interval":
            # The one-sided lower limit is the lower limit of the two-tailed interval at 2 * level - 1
            # (95% -> 90%). Rounded, because in floating point 2 * 0.95 - 1 is 0.8999999999999999.
            compared = COD_Lower(y, x, confidence_level=round(2 * confidence_level - 1, 15))
            limit = standard.high
        else:
            compared = cod
            limit = COD_Cutoff(y, x, property_type, confidence_level)
    _reissue_once(caught)

    if round(cod, 10) < standard.low:
        limits = _cod_limits_text(standard)
        where = f"{standard.source}: {limits}" if standard.source else limits
        _warn_data_quality(
            f"The COD ({cod:.2f}) is below the range for {standard.label} ({where}). Unusually low CODs can "
            "indicate sales chasing or non-representative sales (Section 11.2.5); please check the sales. "
            "COD_Conclusion only tests whether the COD is above the range."
        )
    return _cod_verdict(compared, limit)  # None with too few sales for this test (a warning says why)


# ---------------------------------------------------------------------------
# Other measures of variability (Standard on Ratio Studies 2026, Section
# 8.1.3): the MAPE, the MdAPE and the coefficient of concentration; and
# PE10 / PE20, from the testing of automated valuation models
# ---------------------------------------------------------------------------

def _absolute_percentage_errors(y: Any, x: Any) -> np.ndarray:
    """Each usable sale's absolute percentage error, 100 * |ratio - 1| (after the data checks)."""
    y, x = _sales_pairs(y, x)
    return 100 * np.abs(x / y - 1)


def _check_within(within: float) -> None:
    """Raise a ValueError unless ``within`` is a width in percent of at least 1 (10.0 for +-10%)."""
    if (
        isinstance(within, bool)
        or not isinstance(within, numbers.Real)
        or not math.isfinite(within)
        or within <= 0
    ):
        raise ValueError(
            f"within must be a positive number, in percent (for example 10.0 for +-10%), not {within!r}."
        )
    if within < 1:
        raise ValueError(
            f"within is in percent (10.0 means +-10%), so {within!r} looks like a fraction. "
            f"For +-{within * 100:g}%, use {within * 100:g}."
        )


def _within(ratio: np.ndarray, center: float, within: float) -> np.ndarray:
    """
    True for each ratio within +-``within`` (a fraction: 0.10 for +-10%) of
    ``center``, that is from center x (1 - within) to center x (1 + within),
    edges included. Distances are compared to 10 decimals, so a ratio exactly
    on an edge counts: in floating point, |1.1 - 1| is 0.10000000000000009.
    """
    distance = np.round(np.abs(ratio - center), 10)
    return distance <= round(within * abs(center), 10)


def _percent_within(ratio: np.ndarray, center: float, within: float) -> float:
    """The percentage of the ratios within +-``within`` percent of ``center``: 100 * count / n."""
    return 100 * int(np.count_nonzero(_within(ratio, center, within / 100))) / len(ratio)


def MAPE(y: np.ndarray, x: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE): how far, on average, appraised
    values are from their sale prices.

    The average of each sale's error, the difference between its appraised
    value and its sale price, as a percentage of the sale price. It measures
    the accuracy of a valuation model (IAAO Standard on Ratio Studies 2026,
    Section 8.1.3): an appraised value 10% above or 10% below its sale price
    is an error of 10%.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The MAPE, as a percentage (e.g. 8.5 means 8.5%).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation). An appraised value of 0 is
        kept, as an error of 100%.

    See Also
    --------
    MdAPE : the typical error, less affected by a few very wrong values.
    PE : the percentage of sales appraised within +-10% (or another width)
        of their sale price.
    COD : uniformity, measured around the median ratio instead of the sale
        price.

    Notes
    -----
    Formula. Section 8.1.3 names the MAPE as a "measure of accuracy of
    predictive models" without a formula; this is its usual definition::

        error of a sale = 100 * |appraised value - sale price| / sale price
                        = 100 * |ratio - 1|              (ratio = x / y)
        MAPE            = mean(error)

    For the Standard's 54 example sales (Table 10.1), the MAPE is 23.63%.

    **Level and uniformity together.** The error is measured against the
    sale price, so the MAPE grows both when appraisals are uneven and when
    the overall level is away from 100%. The Standard's 54 example sales
    have a median ratio of 0.8678: even if every sale were appraised at
    exactly that level, the MAPE would be 13.22%. Where values are set at a
    legal level other than market value (for example, one third), the MAPE
    mostly measures that level; use the COD for uniformity and the median
    ratio for the level. One very wrong value moves the MAPE much more than
    the MdAPE.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.MAPE(sales["Sale_Price"], sales["Appraised_Value"])
    >>> rem.MAPE([100_000, 200_000], [110_000, 180_000])  # errors of 10% and 10%: 10.0
    """
    return float(np.mean(_absolute_percentage_errors(y, x)))


def MdAPE(y: np.ndarray, x: np.ndarray) -> float:
    """
    Median Absolute Percentage Error (MdAPE): the typical error of the
    appraised values against their sale prices.

    The middle of the sales' errors (the difference between appraised value
    and sale price, as a percentage of the sale price): half the sales are
    appraised closer to their sale price than this, and half farther. The
    IAAO Standard on Ratio Studies (2026) lists it as a measure of the
    accuracy of predictive models that is "less susceptible to outliers"
    than the MAPE (Section 8.1.3), and notes that "the typical percentage
    error is not the COD but is expressed by the median absolute percentage
    error statistic" (Section 8.1.1).

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float
        The MdAPE, as a percentage (e.g. 6.0 means 6%).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation). An appraised value of 0 is
        kept, as an error of 100%.

    See Also
    --------
    MAPE : the average error.
    PE : the percentage of sales appraised within +-10% (or another width)
        of their sale price.

    Notes
    -----
    Formula. Section 8.1.3 names the MdAPE without a formula; this is its
    usual definition::

        error of a sale = 100 * |appraised value - sale price| / sale price
                        = 100 * |ratio - 1|              (ratio = x / y)
        MdAPE           = median(error); for an even number of sales, the
                          average of the two middle errors (Section 7.1)

    For the Standard's 54 example sales (Table 10.1), the MdAPE is 23.96%.
    Like the MAPE, it measures the level and the uniformity together (see
    ``MAPE``).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.MdAPE(sales["Sale_Price"], sales["Appraised_Value"])
    >>> rem.MdAPE([100_000] * 3, [90_000, 104_000, 120_000])  # errors of 10%, 4%, 20%: 10.0
    """
    return float(np.median(_absolute_percentage_errors(y, x)))


def Coefficient_of_Concentration(y: np.ndarray, x: np.ndarray, within: float = 15.0) -> float:
    """
    Coefficient of concentration: the percentage of sale ratios close to the
    median ratio.

    The share of sales whose ratio is within +-15% (or the width you choose)
    of the median ratio, as a percentage. The higher it is, the more uniform
    the appraisals. The IAAO Standard on Ratio Studies (2026) lists it among
    the "other measures of variability" (Section 8.1.3): "the percentage of
    ratios that are within a certain range (e.g., 15%) of the chosen measure
    of central tendency".

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    within : float, default 15.0
        The width of the range around the median ratio, in percent of the
        median: 15.0 counts the ratios from 0.85 to 1.15 times the median.
        The default, 15%, is the Standard's example (Section 8.1.3 and its
        glossary).

    Returns
    -------
    float
        The percentage of sales whose ratio is within the range (e.g. 62.5
        means 62.5% of the sales).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``within`` is not a number of 1 or more (it is in percent: 15.0
        means +-15%).

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    See Also
    --------
    COD : the average distance of the ratios from the median.
    PE : the same count around 100% (the sale price) instead of the median.
    Sales_Chasing.Ratio_Concentration : the same count at +-2%, to look for
        sales chasing (Appendix B.4).

    Notes
    -----
    Formula (Section 8.1.3 and the glossary)::

        ratio  = appraised value / sale price                    (x / y)
        range  = median ratio * (1 - within / 100)
                 to median ratio * (1 + within / 100)
        result = 100 * (number of ratios in the range) / (number of ratios)

    The range is relative to the median, like the COD, so the result does
    not depend on the overall level: at a median of 0.80, +-15% is 0.68 to
    0.92. Ratios exactly on an edge count as within (distances are compared
    to 10 decimals, because in floating point |0.72 - 0.80| is, for
    example, 0.08000000000000007). The Standard allows any measure of
    central tendency; this function uses the median, its preferred one
    (Section 7.1). For the concentration around 100%, see ``PE``.

    For the Standard's 54 example sales (Table 10.1), 31.5% of the ratios
    (17 of 54) are within +-15% of the median.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.Coefficient_of_Concentration(sales["Sale_Price"], sales["Appraised_Value"])  # +-15%
    >>> rem.Coefficient_of_Concentration(sales["Sale_Price"], sales["Appraised_Value"], within=10)
    """
    _check_within(within)
    y, x = _sales_pairs(y, x)
    ratio = x / y
    return _percent_within(ratio, np.median(ratio), within)


def PE(y: np.ndarray, x: np.ndarray, within: float = 10.0) -> float:
    """
    PE10, PE20: the percentage of sales appraised within +-10% (or +-20%) of
    their sale price.

    The share of sales whose appraised value is within +-10% (or the width
    you choose) of its sale price, as a percentage: PE10 by default, PE20
    with ``within=20``. The higher it is, the more accurate the values.
    PE10 and PE20 come from the testing of automated valuation models
    (AVMs), not from the IAAO Standard on Ratio Studies (2026): they are its
    coefficient of concentration (Section 8.1.3) taken around 100%, the sale
    price, instead of around the median ratio.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    within : float, default 10.0
        The width of the range around each sale price, in percent: 10.0
        (PE10) counts the appraised values from 90% to 110% of their sale
        price, and 20.0 (PE20) from 80% to 120%.

    Returns
    -------
    float
        The percentage of sales appraised within the range (e.g. 62.5 means
        62.5% of the sales).

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``within`` is not a number of 1 or more (it is in percent: 10.0
        means +-10%).

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation).

    See Also
    --------
    Coefficient_of_Concentration : the same count around the median ratio.
    MAPE, MdAPE : the average and the typical size of the errors.

    Notes
    -----
    Formula::

        ratio  = appraised value / sale price                    (x / y)
        result = 100 * (number of sales with |ratio - 1| <= within / 100)
                 / (number of sales)

    so PE10 counts the ratios from 0.90 to 1.10. A value exactly on an edge
    counts as within (distances are compared to 10 decimals, because in
    floating point an appraisal of 110,000 on a sale of 100,000 is
    0.10000000000000009 away). For the Standard's 54 example sales (Table
    10.1), PE10 is 18.5% (10 of 54 sales) and PE20 is 44.4% (24 of 54).

    Like the MAPE, PE measures the level and the uniformity together: at a
    median ratio of 0.80, few values are within +-10% of their sale price
    even when the appraisals are very uniform (see
    ``Coefficient_of_Concentration``).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.PE(sales["Sale_Price"], sales["Appraised_Value"])             # PE10
    >>> rem.PE(sales["Sale_Price"], sales["Appraised_Value"], within=20)  # PE20
    """
    _check_within(within)
    y, x = _sales_pairs(y, x)
    return _percent_within(x / y, 1.0, within)


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

    The Standard on Ratio Studies (2026) says a PRD between 0.98 and 1.03
    indicates good equity (Section 11.2.6). It also says this is the last
    edition in which the PRD is an acceptable measure (footnote 2 of Section
    11), and that the PRD "is highly influenced by value and ratio outliers"
    (Table 8.2). Its preferred measure of vertical equity is the VEI (see
    ``VEI``); the PRB is another (see ``PRB``).

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
        If sales are left out because their value proxy is 0 or negative, or
        if every sale ratio is the same (then there is no bias to measure: the
        slope is 0 and its confidence interval is (0, 0)).

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

    usable_ratio = ratio[usable]
    if np.all(usable_ratio == usable_ratio[0]):
        _warn_data_quality(
            f"Every sale ratio is the same ({usable_ratio[0]:.4f}), so there is no price-related "
            "bias to measure: the PRB is 0 and its confidence interval is (0, 0). Identical ratios "
            "are unusual in real data; please check them (for example, for sales chasing)."
        )

    independent = np.log(value_proxy[usable]) / np.log(2)  # log base 2
    dependent = (usable_ratio - median_ratio) / median_ratio

    return sm.OLS(dependent, sm.add_constant(independent)).fit()


_PRB_LIMIT = 0.05  # Section 11.2.6: a PRB between -0.05 and 0.05 indicates good equity


def _prb_interval(regression: Any, confidence_level: float) -> Tuple[float, float]:
    """
    The two-tailed confidence interval of the PRB slope: (lower, upper).

    ``slope -/+ t * SE(slope)``, where t is the Student-t critical value for
    ``confidence_level`` with n - 2 degrees of freedom (statsmodels'
    ``conf_int``). Row 1 of ``conf_int()`` is the slope; row 0 is the
    intercept.
    """
    # In floating point, 1 - 0.95 is 0.050000000000000044. Rounding gives the
    # intended 0.05 (no confidence level needs 15 decimals), so a 95% interval
    # is exactly the one earlier versions gave.
    alpha = round(1 - confidence_level, 15)
    lower, upper = regression.conf_int(alpha=alpha)[1]
    return lower, upper


def PRB(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Price-Related Bias (PRB) coefficient.

    PRB measures vertical equity: whether sale ratios systematically rise
    (progressivity) or fall (regressivity) as property value increases. It is
    the relative change in sale ratios each time property value doubles. For
    example, a PRB of -0.04 means ratios fall about 4% every time value
    doubles, so higher-value properties are relatively under-appraised. The
    IAAO Standard on Ratio Studies (2026) says a PRB between -0.05 and 0.05
    indicates good equity (Section 11.2.6).

    .. versionchanged:: 0.3.0
       Always returns the coefficient. Earlier versions returned 0.0 when the
       coefficient was not statistically significant (a p-value of 0.05 or
       more), which hid it, most often in small samples. The Standard reports
       the coefficient with its confidence interval (Table 10.2). Use
       ``PRB_Lower`` / ``PRB_Upper`` for the interval and ``PRB_Conclusion``
       for the test.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.

    Returns
    -------
    float or None
        The PRB coefficient, whether or not it is statistically significant.
        ``None`` when 2 or fewer sales can be used, which is too few for a
        regression.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if every sale ratio is the
        same (then the PRB is 0).

    See Also
    --------
    PRB_Lower, PRB_Upper : the confidence interval of the PRB.
    PRB_Conclusion : whether the PRB is significantly outside -0.05 to 0.05.

    Notes
    -----
    The calculation (Standard on Ratio Studies 2026, Glossary and Appendix F,
    Table F.1.1)::

        ratio  = appraised value / sale price                       (x / y)
        proxy  = 0.5 * sale price + 0.5 * appraised value / median ratio
        PRB    = slope of the least-squares line of
                 (ratio - median ratio) / median ratio   on   log2(proxy)

    The value proxy gives equal weight to the sale price and the appraised
    value, so their measurement errors partly offset (Section 8.2). With the
    proxy in log base 2, the slope is the change in ratios per doubling of
    value. The full formula is in ``_prb_regression``.

    The PRB is a point estimate. Whether it is statistically significantly
    outside -0.05 to 0.05 is decided by its confidence interval
    (``PRB_Conclusion``). For the Standard's 54 example sales (Table 10.1),
    the PRB is 0.1871 with a t-value of 3.113 (Table 10.2).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> prb = rem.PRB(sales["Sale_Price"], sales["Appraised_Value"])
    """
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None
    return regression.params[1]


def PRB_Lower(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> Optional[float]:
    """
    Lower limit of the PRB's two-tailed confidence interval (90% by default).

    See ``PRB`` for what the Price-Related Bias measures.

    .. versionchanged:: 0.3.0
       New ``confidence_level`` argument, 0.90 by default: the level the
       Standard on Ratio Studies (2026) uses for compliance (Section 11.1.2)
       and in its demonstration report (Table 10.2). Earlier versions always
       gave a 95% interval; ``confidence_level=0.95`` gives exactly the same
       result.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1.

    Returns
    -------
    float or None
        The lower confidence limit of the PRB. ``None`` when 2 or fewer sales
        can be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if every sale ratio is the
        same (then the interval is (0, 0)).

    Notes
    -----
    The interval::

        PRB -/+ t * SE(PRB)

    where SE is the standard error of the regression slope and t is the
    two-tailed Student-t critical value for ``confidence_level`` with n - 2
    degrees of freedom (about 1.675 for 90% and the Standard's 54 example
    sales). For those sales the 90% interval is (0.0865, 0.2878). The
    Standard prints the lower limit as 0.0864 because it used the t-table
    value 1.675; the exact value, 1.6747, gives 0.0865.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> lower = rem.PRB_Lower(sales["Sale_Price"], sales["Appraised_Value"])
    >>> lower_95 = rem.PRB_Lower(sales["Sale_Price"], sales["Appraised_Value"],
    ...                          confidence_level=0.95)
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None
    lower, _ = _prb_interval(regression, confidence_level)
    return lower


def PRB_Upper(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> Optional[float]:
    """
    Upper limit of the PRB's two-tailed confidence interval (90% by default).

    See ``PRB`` for what the Price-Related Bias measures, and ``PRB_Lower``
    for how the interval is calculated.

    .. versionchanged:: 0.3.0
       New ``confidence_level`` argument, 0.90 by default: the level the
       Standard on Ratio Studies (2026) uses for compliance (Section 11.1.2)
       and in its demonstration report (Table 10.2). Earlier versions always
       gave a 95% interval; ``confidence_level=0.95`` gives exactly the same
       result.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1.

    Returns
    -------
    float or None
        The upper confidence limit of the PRB. ``None`` when 2 or fewer sales
        can be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if every sale ratio is the
        same (then the interval is (0, 0)).

    Notes
    -----
    For the Standard's 54 example sales (Table 10.1), the 90% interval is
    (0.0865, 0.2878); Table 10.2 prints the lower limit as 0.0864 (see
    ``PRB_Lower``).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> upper = rem.PRB_Upper(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None
    _, upper = _prb_interval(regression, confidence_level)
    return upper


def PRB_Conclusion(y: np.ndarray, x: np.ndarray, confidence_level: float = 0.90) -> Optional[str]:
    """
    Does the PRB meet the Standard? ``'PASS'`` or ``'FAIL'``.

    ``'FAIL'`` only when the PRB's whole confidence interval lies outside
    -0.05 to 0.05: the PRB is then statistically significantly outside the
    range the Standard on Ratio Studies (2026) calls good equity. Otherwise
    ``'PASS'``. See ``PRB`` for what the Price-Related Bias measures.

    .. versionchanged:: 0.3.0
       Rewritten to follow the Standard's confidence-interval principle (see
       Notes), with a ``confidence_level`` argument (0.90 by default). Earlier
       versions failed a PRB that differed from 0 at the 95% level and was at
       least 0.05 in size. A PRB right at the limit then failed about half the
       time, although the evidence could not show it was outside the range.
       When every sale ratio is the same, the result is now ``'PASS'`` (there
       is no bias), with a warning; it used to be ``'FAIL'``.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    confidence_level : float, default 0.90
        Confidence level of the two-tailed interval, between 0 and 1.

    Returns
    -------
    str or None
        ``'PASS'`` or ``'FAIL'``. ``None`` when 2 or fewer sales can be used.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers, or if
        ``confidence_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), or if every sale ratio is the
        same.

    Notes
    -----
    The decision rule::

        lower, upper = the PRB's confidence interval (see PRB_Lower)
        'FAIL'  if lower > 0.05 or upper < -0.05
        'PASS'  otherwise

    The limits are compared after rounding to 10 decimal places, so a limit
    of exactly 0.05 counts as inside the range.

    Why the rule is written this way: the Standard gives the PRB's acceptable
    range, -0.05 to 0.05 (Section 11.2.6; Appendix F, Table F.1.1), and lists
    the PRB as having built-in statistical testing (Table 8.1), but it does
    not spell out a test for the PRB. This rule applies the principle the
    Standard uses for compliance everywhere else:

    * "A conclusion of noncompliance requires a high degree of confidence;
      thus, a 90 percent (two-tailed) or 95 percent (one-tailed) confidence
      level should be used, except for small or highly variable samples"
      (Section 11.1.2);
    * compliance "should be evaluated using confidence intervals and other
      measures of statistical significance" (Section 13.6.2);
    * the VEI fails only when it is "statistically significantly outside" its
      range (Section 11.2.6 and Appendix E, Table E.7.1), which ``VEI``
      implements the same way.

    So ``'PASS'`` means the evidence does not show the PRB is outside the
    range; it is not proof that the PRB is inside it. With few sales the
    interval is wide, and a PRB beyond the limit can still pass: look at the
    PRB and its interval too.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.PRB_Conclusion(sales["Sale_Price"], sales["Appraised_Value"])
    """
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")
    y, x = _sales_pairs(y, x)
    regression = _prb_regression(y, x)
    if regression is None:
        return None
    return _prb_verdict(*_prb_interval(regression, confidence_level))


def _prb_verdict(lower: float, upper: float) -> str:
    """'FAIL' when the PRB's whole confidence interval lies outside +-0.05, otherwise 'PASS'."""
    # Rounded, so floating-point arithmetic can't push a limit of exactly 0.05 past it.
    lower, upper = round(lower, 10), round(upper, 10)
    if lower > _PRB_LIMIT or upper < -_PRB_LIMIT:
        return 'FAIL'
    return 'PASS'


class _CodStandard(NamedTuple):
    """A COD range for the summary's note: its limits, where it comes from, and what it covers."""

    low: float
    high: float
    source: Optional[str]  # "Table 11.1" or "Table 11.2"; None for the user's own range
    label: str


# The COD ranges of the Standard on Ratio Studies (2026): Table 11.1 (ratio study uniformity
# standards) and Table 11.2 (blighted, distressed and emerging market areas). Keys are the names
# Ratio_Study_Report's (and DOR_SUMMARY_Statistics') property_type accepts; its docstring lists them all.
_COD_STANDARDS = {
    "residential": _CodStandard(5.0, 15.0, "Table 11.1", "residential improved"),
    "residential_homogeneous": _CodStandard(
        4.0, 10.0, "Table 11.1",
        "residential improved, newer or more homogeneous market areas such as condominiums"),
    "residential_rural": _CodStandard(
        10.0, 20.0, "Table 11.1",
        "residential improved, rural, seasonally occupied housing, or manufactured homes"),
    "multi_unit": _CodStandard(7.0, 20.0, "Table 11.1", "multi-unit apartments"),
    "multi_unit_rural": _CodStandard(10.0, 25.0, "Table 11.1", "multi-unit apartments, rural jurisdictions"),
    "commercial": _CodStandard(10.0, 25.0, "Table 11.1", "commercial, industrial, retail, warehouse, etc."),
    "commercial_condominium": _CodStandard(5.0, 15.0, "Table 11.1", "commercial/industrial condominiums"),
    "commercial_less_active": _CodStandard(
        10.0, 30.0, "Table 11.1",
        "commercial, industrial, retail, warehouse, etc., less active markets with small sample size"),
    "vacant_land": _CodStandard(10.0, 20.0, "Table 11.1", "vacant land"),
    "vacant_land_less_active": _CodStandard(
        10.0, 25.0, "Table 11.1", "vacant land, less active markets with small sample size"),
    "vacant_land_rural": _CodStandard(10.0, 30.0, "Table 11.1", "vacant land, rural jurisdictions"),
    "agricultural": _CodStandard(10.0, 25.0, "Table 11.1", "agricultural land"),
    "residential_distressed": _CodStandard(
        10.0, 25.0, "Table 11.2", "residential improved, distressed or blighted market areas"),
    "residential_emerging": _CodStandard(10.0, 35.0, "Table 11.2", "residential improved, emerging market areas"),
    "multi_unit_distressed": _CodStandard(
        10.0, 30.0, "Table 11.2", "multi-family, distressed or blighted market areas"),
    "multi_unit_emerging": _CodStandard(10.0, 35.0, "Table 11.2", "multi-family, emerging market areas"),
    "commercial_distressed": _CodStandard(
        10.0, 35.0, "Table 11.2",
        "commercial, industrial, retail, warehouse, etc., distressed or blighted market areas"),
    "commercial_emerging": _CodStandard(
        10.0, 40.0, "Table 11.2", "commercial, industrial, retail, warehouse, etc., emerging market areas"),
    "vacant_land_distressed": _CodStandard(10.0, 30.0, "Table 11.2", "vacant land, distressed or blighted market areas"),
    "vacant_land_emerging": _CodStandard(10.0, 45.0, "Table 11.2", "vacant land, emerging market areas"),
    # Table 11.2 has one agricultural row for all three market conditions.
    "agricultural_distressed": _CodStandard(
        10.0, 45.0, "Table 11.2", "agricultural land, distressed, blighted or emerging market areas"),
    "agricultural_emerging": _CodStandard(
        10.0, 45.0, "Table 11.2", "agricultural land, distressed, blighted or emerging market areas"),
}


def _simplified_name(name: str) -> str:
    """Lower case, without spaces, hyphens or underscores: 'Vacant Land' -> 'vacantland'."""
    return "".join(character for character in name.lower() if character not in " -_")


_COD_STANDARD_NAMES = {_simplified_name(name): name for name in _COD_STANDARDS}


def _cod_standard(property_type: Any) -> Optional[_CodStandard]:
    """
    The COD range for ``Ratio_Study_Report``'s ``property_type``, or None
    for no COD note ("none", or None).

    Raises
    ------
    ValueError
        For an unknown name (the message lists every accepted name), or a
        range that is not two numbers with 0 <= low < high.
    """
    if property_type is None:
        return None
    if isinstance(property_type, str):
        simplified = _simplified_name(property_type)
        if simplified == "none":
            return None
        if simplified in _COD_STANDARD_NAMES:
            return _COD_STANDARDS[_COD_STANDARD_NAMES[simplified]]
    elif isinstance(property_type, (tuple, list)) and len(property_type) == 2:
        low, high = property_type
        if (
            all(isinstance(limit, numbers.Real) and not isinstance(limit, bool) for limit in (low, high))
            and math.isfinite(low) and math.isfinite(high)
            and 0 <= low < high
        ):
            return _CodStandard(float(low), float(high), None, "your local standard")
        raise ValueError(
            "property_type: a local COD range must be two numbers (low, high) with 0 <= low < high, "
            f"for example (5.0, 12.0), not {property_type!r}."
        )
    names = ", ".join(f"'{name}'" for name in ["none", *_COD_STANDARDS])
    raise ValueError(
        f"property_type must be one of {names}, or a (low, high) COD range for a local standard; "
        f"got {property_type!r}."
    )


def _cod_limit_text(limit: float) -> str:
    """A COD limit as the reports print it: 5 -> '5.0', 12.25 -> '12.25'."""
    return f"{limit:.1f}" if round(limit, 1) == limit else f"{limit:g}"


def _cod_limits_text(standard: _CodStandard) -> str:
    """A COD range as the reports print it: '5.0 to 15.0' (a limit like 12.25 keeps its decimals)."""
    return f"{_cod_limit_text(standard.low)} to {_cod_limit_text(standard.high)}"


def _cod_result_text(conclusion: Optional[str], standard: Optional[_CodStandard], confidence_level: float) -> str:
    """
    The summary's COD verdict, after the COD's note: nothing without a
    standard; ' <-> COD RESULT: FAIL (90% CI entirely above 15.0)', its PASS
    form, or ' <-> COD RESULT: not calculated' when there are too few sales.
    """
    if standard is None:
        return ""
    if conclusion is None:
        return " <-> COD RESULT: not calculated"
    relation = "entirely above" if conclusion == 'FAIL' else "not entirely above"
    return (
        f" <-> COD RESULT: {conclusion} ({_percent(confidence_level)} CI {relation} "
        f"{_cod_limit_text(standard.high)})"
    )


def _cod_note(cod: float, standard: Optional[_CodStandard]) -> str:
    """
    The text printed after the COD: nothing without a standard; the range when
    the COD is within it; a NOTE when it is above or below it (Section 11.2.5:
    a COD below the range is "a diagnostic warning").
    """
    if standard is None:
        return ""

    limits = _cod_limits_text(standard)
    where = f"{standard.source}: {limits}" if standard.source else limits
    cod = round(cod, 10)  # a COD exactly at a limit counts as within, despite floating-point noise
    if cod < standard.low:
        return (
            f" <- NOTE: BELOW the range for {standard.label} ({where}). Unusually low CODs can "
            "indicate sales chasing or non-representative sales (Section 11.2.5)."
        )
    if cod > standard.high:
        return f" <- NOTE: ABOVE the range for {standard.label} ({where})"
    if standard.source:
        return f" ({standard.source}, {standard.label}: {limits})"
    return f" ({standard.label}: {limits})"


def DOR_SUMMARY_Statistics(
    y: np.ndarray,
    x: np.ndarray,
    return_results: bool = False,
    property_type: Union[str, Tuple[float, float], None] = "none",
) -> Optional[dict]:
    """
    Print a summary of the main ratio-study statistics.

    .. deprecated:: 0.3.0
       ``DOR_SUMMARY_Statistics`` will be removed in pyRealEstate 0.4.0.
       Use ``Ratio_Study_Report``, which returns the same statistics and
       more (Table 10.2's sales totals and averages, the PRB's t-value, the
       MAPE, MdAPE, coefficient of concentration, PE10 and PE20) as a table:
       print it, or show it in a notebook. Each key of the dictionary below
       is one of its attributes in lower case (``'COD_Lower'`` is
       ``cod_lower``; ``'Weighted_Mean'`` is ``weighted_mean_ratio``).

    Calculates the three measures of the level of valuation (the median,
    mean and weighted mean sale ratios) and the COD, each with its 90%
    confidence interval; the PRD; the PRB (with its 90% confidence interval
    and PASS / FAIL conclusion); the Vertical Equity Indicator (VEI, with
    its 90% confidence interval and conclusion); and whether the ratios are
    normally distributed (``Normality_Test``), and prints them. A PRD
    outside 0.98 to 1.03 is noted. When you name a property type
    (``property_type``), a COD outside its range is noted, and the COD's
    verdict, PASS or FAIL, is shown (``COD_Conclusion``). The VEI is the
    Standard's preferred measure of vertical equity (Standard on Ratio
    Studies 2026, Section 8.2.1).

    .. versionchanged:: 0.3.0
       The summary opens with the median and mean sale ratios, beside the
       weighted mean, as the Standard's demonstration report does (Table
       10.2), and ends with the VEI and a test of normality, as that report
       does. The median, mean and weighted mean ratios and the COD show
       their 90% confidence intervals (Appendix D), and the PRB line shows
       its own. The dictionary gains the keys ``'VEI'``, ``'VEI_Lower'``,
       ``'VEI_Upper'``, ``'VEI_Conclusion'``, ``'Median_Ratio'``,
       ``'Mean_Ratio'``, the eight interval keys, ``'COD_Conclusion'``,
       ``'Normality_Conclusion'`` and ``'Normality_p_value'``, added at the
       end so the older keys keep their order.

       New ``property_type`` argument: the COD is compared with the
       Standard's range for that property type and market (Tables 11.1 and
       11.2), and tested against its maximum. By default (``"none"``) the COD
       is printed without a note or a verdict.
       Earlier versions noted every COD above 10 as moderately high and above
       15 as abnormally high, thresholds the 2026 Standard does not use.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    return_results : bool, default False
        If True, also return the statistics as a dictionary, so they can be
        used in further calculations or saved.
    property_type : str or (float, float), default "none"
        The COD standard to compare the COD with. By default (``"none"``, or
        None) the COD is printed without a note or a verdict, so no property
        type is assumed. Name the property type and market of the study to
        compare the COD with the Standard's range (Standard on Ratio Studies
        2026, Tables 11.1 and 11.2) and test it against the range's maximum,
        or give your own ``(low, high)`` range for a local standard. Names may use any capitalization, and spaces, hyphens
        or underscores ("Vacant Land" works). Every accepted name, with its
        COD range and the Standard's row:

        Table 11.1, ratio study uniformity standards::

            "residential"               5.0 to 15.0   Residential improved
            "residential_homogeneous"   4.0 to 10.0   - newer or more homogeneous
                                                        market areas such as condominiums
            "residential_rural"        10.0 to 20.0   - rural, seasonally occupied
                                                        housing, or manufactured homes
            "multi_unit"                7.0 to 20.0   Multi-unit apartments
            "multi_unit_rural"         10.0 to 25.0   - rural jurisdictions
            "commercial"               10.0 to 25.0   Commercial, industrial, retail,
                                                      warehouse, etc.
            "commercial_condominium"    5.0 to 15.0   - commercial/industrial condominiums
            "commercial_less_active"   10.0 to 30.0   - less active markets with small
                                                        sample size
            "vacant_land"              10.0 to 20.0   Vacant land
            "vacant_land_less_active"  10.0 to 25.0   - less active markets with small
                                                        sample size
            "vacant_land_rural"        10.0 to 30.0   - rural jurisdictions
            "agricultural"             10.0 to 25.0   Agricultural land

        Table 11.2, blighted, distressed, and emerging market areas::

            "residential_distressed"   10.0 to 25.0   Residential improved, distressed
                                                      or blighted market areas
            "residential_emerging"     10.0 to 35.0   Residential improved, emerging
            "multi_unit_distressed"    10.0 to 30.0   Multi-family, distressed or blighted
            "multi_unit_emerging"      10.0 to 35.0   Multi-family, emerging
            "commercial_distressed"    10.0 to 35.0   Commercial, industrial, ...,
                                                      distressed or blighted
            "commercial_emerging"      10.0 to 40.0   Commercial, industrial, ..., emerging
            "vacant_land_distressed"   10.0 to 30.0   Vacant land, distressed or blighted
            "vacant_land_emerging"     10.0 to 45.0   Vacant land, emerging
            "agricultural_distressed"  10.0 to 45.0   Agricultural land (the table has one
            "agricultural_emerging"    10.0 to 45.0   row for distressed, blighted and
                                                      emerging market areas)

        And::

            "none"                     (the default) no COD note
            (low, high)                your local standard, e.g. (5.0, 12.0)

    Returns
    -------
    dict or None
        When ``return_results`` is True, a dictionary with the keys
        ``'Weighted_Mean'``, ``'COD'``, ``'PRD'``, ``'PRB'``, ``'PRB_Lower'``,
        ``'PRB_Upper'``, ``'PRB_Conclusion'``, ``'VEI'``, ``'VEI_Lower'``,
        ``'VEI_Upper'``, ``'VEI_Conclusion'``, ``'Median_Ratio'``,
        ``'Mean_Ratio'``, ``'Median_Ratio_Lower'``, ``'Median_Ratio_Upper'``,
        ``'Mean_Ratio_Lower'``, ``'Mean_Ratio_Upper'``,
        ``'Weighted_Mean_Lower'``, ``'Weighted_Mean_Upper'``, ``'COD_Lower'``,
        ``'COD_Upper'``, ``'COD_Conclusion'``, ``'Normality_Conclusion'`` and
        ``'Normality_p_value'``, in that order. The VEI values are in
        percent, and NaN with fewer than 20 sales; the COD and its limits are
        in percent; an interval that can't be calculated is NaN;
        ``'COD_Conclusion'`` is ``'PASS'`` or ``'FAIL'`` with a
        ``property_type``, and None without one (or with too few sales);
        ``'Normality_Conclusion'`` and ``'Normality_p_value'`` are
        ``Normality_Test``'s conclusion and p-value (NaN when the test can't
        be done). New keys are only added at the end, so the order of the
        older keys does not change. Otherwise ``None``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; or if
        ``property_type`` is not an accepted name or a range of two numbers
        with 0 <= low < high. ``property_type`` is checked first, so nothing
        is calculated or printed when it is wrong.

    Warns
    -----
    FutureWarning
        Always, to say that ``DOR_SUMMARY_Statistics`` is deprecated (use
        ``Ratio_Study_Report``). Python shows it once for each line of your
        code that calls it.
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), if there are too few sales for
        the VEI (it needs 20), if a confidence interval can't be calculated
        (for example, the median's and the COD's need 5 sales), or if the
        test of normality can't be done (3 sales) or is approximate (more
        than 5,000 sales; see ``Normality_Test``). Each statistic checks the
        data, but each problem is reported once.

    Notes
    -----
    Each statistic is calculated by the function of the same name in this
    module (``Median_Ratio``, ``Mean_Ratio``, ``Weighted_Mean_Ratio``,
    ``COD``, ``PRD``, ``PRB``, ``PRB_Lower``, ``PRB_Upper``,
    ``PRB_Conclusion``, ``VEI``). See their documentation for the formulas
    (the level measures: Sections 7.1 to 7.3). The median comes first: the
    Standard calls it "the generally preferred measure of central tendency
    for evaluating overall valuation level" (Section 7.1). For the 54 sales
    of the Standard's Table 10.1, the summary gives Table 10.2's values: a
    median of 0.8678, a mean of 0.8938 and a weighted mean of 0.9003.

    Each confidence interval is calculated once, by the same calculation as
    the functions ``Median_Ratio_Lower`` / ``Median_Ratio_Upper``,
    ``Mean_Ratio_Lower`` / ``_Upper``, ``Weighted_Mean_Lower`` / ``_Upper``
    and ``COD_Lower`` / ``_Upper`` (Appendix D), and printed after its
    statistic as ``<-> 90% CI: lower to upper``, or ``not calculated`` when
    there are too few sales for it (the dictionary then holds NaN). For the
    Standard's 54 example sales the intervals are Table 10.2's: median
    (0.7867, 0.9840), mean (0.8350, 0.9526), weighted mean (0.8359,
    0.9647), COD (21.878%, 28.618%).

    The intervals, the PRB and the VEI use their defaults: 90% confidence
    (the Standard's level, Section 11.1.2) and, for the VEI, R7 ranking, as
    in the Standard's worked example. Call them directly for other options;
    ``VEI`` also has a group table and a full report. Like ``VEI``, the
    summary's VEI leaves out sales with an appraised value of 0 or less,
    which the other statistics keep.

    The printed notes:

    * **Level of valuation:** none. The Standard's range is 0.90 to 1.10
      (Section 11.1 and the footnote of Table 11.1), but it measures the
      level against the legally required assessment level (footnote 1 of
      Section 11.2.1), which the summary cannot know: a jurisdiction that
      assesses at one third of market value would be flagged on every run.
      A point estimate is also "only an indication, not proof" of the level
      (Section 11.1): compare the confidence intervals with your legal level.
    * **COD**, only when ``property_type`` is given. A COD above the range is
      noted, and so is one below it: the Standard says CODs below the minimum
      "should be treated as a diagnostic warning", for example of sales
      chasing, non-representative samples or model overfitting (Section
      11.2.5). Within the range, the range is shown. A COD exactly at a limit
      counts as within. The Standard adds that its ranges "are provided for
      guidance only and may not represent jurisdictional requirements", that
      they apply to ratio studies based on sales (not on independent
      appraisals, which have lower CODs), and that a COD is a point estimate,
      not proof of non-compliance (Section 11.2).
    * **COD RESULT**, only when ``property_type`` is given, after the note:
      ``COD_Conclusion``'s verdict with its defaults, the interval test at
      95% one-sided, which is the printed 90% interval. 'FAIL' only when
      that interval lies entirely above the range's maximum, shown as
      ``COD RESULT: FAIL (90% CI entirely above 15.0)``; otherwise 'PASS'
      (``not entirely above``); ``not calculated`` when there are too few
      sales for the interval. A COD below the range is a note, not a
      failure. For the Standard's 54 example sales as residential property,
      the interval, 21.878 to 28.618, gives 'FAIL'. Call ``COD_Conclusion``
      for the tolerance test of Appendix D.8 (``method="tolerance"``).
    * **Normal distribution of ratios**, the last line, where Table 10.2
      reports it: ``Normality_Test``'s conclusion with its defaults (the
      Shapiro-Wilk test at 0.05) and its p-value, e.g.
      ``Normal distribution of ratios (Shapiro-Wilk, 0.05): No significant
      departure from normality (p = 0.1962)`` for the Standard's 54 example
      sales. (Table 10.2 says "Reject" for those sales; it does not say
      which test it used, and none of the three tests of Table 6.3 gives
      that result.) When the ratios are not normally distributed, the
      mean-based statistics (the mean, weighted mean and their intervals)
      are less reliable than the median and the COD (Section 6.2). Call
      ``Normality_Test`` for the other tests and the ratios' skewness and
      kurtosis.
    * **PRD** below 0.98 is noted as abnormally low, and above 1.03 as
      abnormally high: the Standard's range for good equity is 0.98 to 1.03
      (Section 11.2.6). It also says this is the last edition in which the
      PRD is an acceptable measure (footnote 2 of Section 11).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.DOR_SUMMARY_Statistics(sales["Sale_Price"], sales["Appraised_Value"])
    >>> rem.DOR_SUMMARY_Statistics(sales["Sale_Price"], sales["Appraised_Value"],
    ...                            property_type="Residential")   # Table 11.1: 5.0 to 15.0
    >>> rem.DOR_SUMMARY_Statistics(sales["Sale_Price"], sales["Appraised_Value"],
    ...                            property_type=(5.0, 12.0))     # a local standard
    >>> results = rem.DOR_SUMMARY_Statistics(
    ...     sales["Sale_Price"], sales["Appraised_Value"], return_results=True
    ... )
    >>> results["Median_Ratio"], results["COD"], results["VEI_Conclusion"]
    >>> results["COD_Lower"], results["COD_Upper"]
    >>> rem.DOR_SUMMARY_Statistics(sales["Sale_Price"], sales["Appraised_Value"],
    ...                            return_results=True, property_type="Residential")["COD_Conclusion"]
    """
    _warn_deprecated(
        "DOR_SUMMARY_Statistics",
        "Ratio_Study_Report",
        "which returns the same statistics and more as a table (print it, or show it in a notebook)",
    )
    cod_standard = _cod_standard(property_type)  # checked first: nothing is printed if it is wrong
    confidence_level = 0.90  # the Standard's level (Section 11.1.2), as for the PRB and the VEI

    # Calculate each statistic once, then print. Each statistic checks the
    # data itself, so collect their data-quality warnings and report each
    # distinct problem once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataQualityWarning)
        median_ratio = Median_Ratio(y, x)
        mean_ratio = Mean_Ratio(y, x)
        weighted_mean = Weighted_Mean_Ratio(y, x)
        cod = COD(y, x)
        # Each interval's lower and upper limit from one calculation (the helper its _Lower and
        # _Upper functions share)
        median_lower, median_upper = _median_ratio_interval(y, x, confidence_level)
        mean_lower, mean_upper = _mean_ratio_interval(y, x, confidence_level)
        weighted_mean_lower, weighted_mean_upper = _weighted_mean_interval(y, x, confidence_level)
        cod_lower, cod_upper = _cod_interval(y, x, confidence_level)
        prd = PRD(y, x)
        prb = PRB(y, x)
        prb_lower = PRB_Lower(y, x)
        prb_upper = PRB_Upper(y, x)
        prb_conclusion = PRB_Conclusion(y, x)
        vei = VEI(y, x)
        normality = Normality_Test(y, x)  # its defaults: Shapiro-Wilk at 0.05
    _reissue_once(caught)

    # The COD's verdict against the property type's maximum: COD_Conclusion's default test, the interval
    # test at 95% one-sided, whose lower limit is the 90% interval's (calculated above, so not again)
    cod_conclusion = None if cod_standard is None else _cod_verdict(cod_lower, cod_standard.high)

    # The three measures of the level of valuation, the Standard's preferred one first
    print(f"Median Ratio: {median_ratio}{_interval_text(median_lower, median_upper, confidence_level)}\n")
    print(f"Mean Ratio: {mean_ratio}{_interval_text(mean_lower, mean_upper, confidence_level)}\n")
    print(f"Weighted Mean: {weighted_mean}"
          f"{_interval_text(weighted_mean_lower, weighted_mean_upper, confidence_level)}\n")

    print(
        f"COD: {cod}{_interval_text(cod_lower, cod_upper, confidence_level)}{_cod_note(cod, cod_standard)}"
        f"{_cod_result_text(cod_conclusion, cod_standard, confidence_level)}\n"
    )

    if prd < 0.98:
        print(f"PRD: {prd} <- NOTE THIS IS ABNORMALLY LOW\n")
    elif prd > 1.03:
        print(f"PRD: {prd} <- NOTE THIS IS ABNORMALLY HIGH\n")
    else:
        print(f"PRD: {prd}\n")

    print(
        f"PRB: {prb} <-> "
        f"PRB 90% Lower Bound: {prb_lower} <-> "
        f"PRB 90% Upper Bound: {prb_upper} <-> "
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

    # Last, where the Standard's demonstration report (Table 10.2) puts it
    print(f"\n{_normality_text(normality)}")

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
            # Added in 0.3.0, after the older keys so their order does not change
            'Median_Ratio': median_ratio,
            'Mean_Ratio': mean_ratio,
            'Median_Ratio_Lower': median_lower,
            'Median_Ratio_Upper': median_upper,
            'Mean_Ratio_Lower': mean_lower,
            'Mean_Ratio_Upper': mean_upper,
            'Weighted_Mean_Lower': weighted_mean_lower,
            'Weighted_Mean_Upper': weighted_mean_upper,
            'COD_Lower': cod_lower,
            'COD_Upper': cod_upper,
            'COD_Conclusion': cod_conclusion,
            'Normality_Conclusion': normality.conclusion,
            'Normality_p_value': normality.p_value,
        }
    return None


# ---------------------------------------------------------------------------
# Small helpers shared by the reports and option checks below
# ---------------------------------------------------------------------------

def _percent(fraction: float) -> str:
    """0.9 -> '90%', 0.975 -> '97.5%'."""
    return f"{fraction * 100:g}%"


def _normality_text(result: "NormalityTestResult") -> str:
    """
    The summary's normality line, as Table 10.2 reports it: 'Normal distribution
    of ratios (Shapiro-Wilk, 0.05): <conclusion> (p = 0.1962)', without the
    p-value when the test can't be done.
    """
    p_value = "" if math.isnan(result.p_value) else f" (p {_format_p_value(result.p_value)})"
    return f"Normal distribution of ratios ({result.method}, {result.significance_level:g}): {result.conclusion}{p_value}"


def _interval_text(lower: float, upper: float, confidence_level: float) -> str:
    """' <-> 90% CI: <lower> to <upper>' for a summary line, or ' <-> 90% CI: not calculated' (NaN limits)."""
    if math.isnan(lower):
        return f" <-> {_percent(confidence_level)} CI: not calculated"
    return f" <-> {_percent(confidence_level)} CI: {lower} to {upper}"


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
# The ratio study report: every statistic of a ratio study in one table
#
# Source: IAAO Standard on Ratio Studies (2026), Table 10.2 ("Demonstration
# Ratio Study Report"), Section 10.1.2 (what a report shows) and Section 8.1.3
# (the MAPE, MdAPE and coefficient of concentration).
# ---------------------------------------------------------------------------

_NOT_CALCULATED = "not calculated"
_PRB_LABEL = "PRB (t-value)"
_REPORT_SOURCE = (
    "IAAO Standard on Ratio Studies (2026): Table 10.2's statistics; the MAPE, MdAPE and coefficient of "
    "concentration, Section 8.1.3. PE10 and PE20 are from AVM testing, not the Standard."
)


class _ReportRow(NamedTuple):
    """One statistic of the report: its section and label, its unrounded numbers, and the three cells shown."""

    section: str
    statistic: str
    value: float  # NaN when not calculated
    lower: float  # the confidence interval; NaN when there is none
    upper: float
    value_text: str
    interval_text: str
    result: str


@dataclass(frozen=True, eq=False, repr=False)
class RatioStudyReport:
    """
    The ratio study report: every statistic of a ratio study, shown as a table.

    Returned by ``Ratio_Study_Report``. In a notebook (Jupyter, VS Code,
    Colab) it shows itself as a formatted table; ``print(report)`` shows the
    same table as aligned text. Each statistic is also an attribute, at full
    precision, named like the function that calculates it, in lower case.
    The table rounds each value as the IAAO Standard on Ratio Studies (2026)
    does in Table 10.2.

    .. versionadded:: 0.3.0

    Attributes
    ----------
    n_sales, n_left_out : int
        The number of sales used, and of sales left out because they cannot
        form a sale ratio (see "Data checks" in the module documentation).
    total_appraised_value, total_sale_price : float
        The totals of the sales' appraised values and sale prices (Table
        10.2).
    average_appraised_value, average_sale_price : float
        Their averages (Table 10.2).
    median_ratio, median_ratio_lower, median_ratio_upper : float
        The median ratio and its confidence interval (Section 7.1, Appendix
        D.2 and D.4).
    mean_ratio, mean_ratio_lower, mean_ratio_upper : float
        The mean ratio and its confidence interval (Section 7.2, Appendix
        D.1).
    weighted_mean_ratio, weighted_mean_lower, weighted_mean_upper : float
        The weighted mean ratio and its confidence interval (Section 7.3,
        Appendix D.3).
    cod, cod_lower, cod_upper : float
        The COD and its confidence interval, in percent (Section 8.1.1,
        Appendix D.6).
    cod_conclusion : str or None
        'PASS' or 'FAIL' against the COD standard of ``property_type``
        (``COD_Conclusion``); None without one, or with too few sales.
    cod_standard : tuple or None
        The COD range compared with, as (low, high, source, label), for
        example (5.0, 15.0, 'Table 11.1', 'residential improved'); None
        without a ``property_type``.
    coefficient_of_concentration : float
        The percentage of ratios within +-15% of the median ratio (Section
        8.1.3).
    mape, mdape : float
        The mean and median absolute percentage errors against the sale
        prices, in percent (Section 8.1.3).
    pe10, pe20 : float
        The percentages of sales appraised within +-10% and +-20% of their
        sale prices (AVM practice).
    vei, vei_lower, vei_upper : float
        The VEI and its confidence interval, in percent (Appendix E); NaN
        with fewer than 20 sales.
    vei_conclusion : str
        ``VEI_Conclusion``'s verdict.
    prb, prb_t_value, prb_lower, prb_upper : float or None
        The PRB coefficient, its t-value and its confidence interval
        (Section 11.1.2, Table 10.2); None with 2 or fewer sales. The
        t-value is NaN when every ratio is the same.
    prb_conclusion : str or None
        ``PRB_Conclusion``'s verdict, 'PASS' or 'FAIL'.
    prd : float
        The PRD (Section 11.2.6).
    normality_method, normality_significance_level, normality_p_value, normality_conclusion
        ``Normality_Test``'s test (Shapiro-Wilk), its significance level
        (0.05), p-value and conclusion.
    confidence_level : float
        The two-tailed confidence level of the intervals.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> report = rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"])
    >>> report                    # in a notebook: the table
    >>> print(report)             # anywhere: the table as text
    >>> report.cod, report.cod_lower, report.pe10
    >>> report.to_frame().to_excel("ratio_study.xlsx", index=False)
    """

    n_sales: int
    n_left_out: int
    total_appraised_value: float
    total_sale_price: float
    average_appraised_value: float
    average_sale_price: float
    median_ratio: float
    median_ratio_lower: float
    median_ratio_upper: float
    mean_ratio: float
    mean_ratio_lower: float
    mean_ratio_upper: float
    weighted_mean_ratio: float
    weighted_mean_lower: float
    weighted_mean_upper: float
    cod: float
    cod_lower: float
    cod_upper: float
    cod_conclusion: Optional[str]
    cod_standard: Optional[_CodStandard]
    coefficient_of_concentration: float
    mape: float
    mdape: float
    pe10: float
    pe20: float
    vei: float
    vei_lower: float
    vei_upper: float
    vei_conclusion: str
    prb: Optional[float]
    prb_t_value: Optional[float]
    prb_lower: Optional[float]
    prb_upper: Optional[float]
    prb_conclusion: Optional[str]
    prd: float
    normality_method: str
    normality_significance_level: float
    normality_p_value: float
    normality_conclusion: str
    confidence_level: float

    def __repr__(self) -> str:
        return (
            f"RatioStudyReport(n_sales={self.n_sales}, median_ratio={self.median_ratio:.4f}, cod={self.cod:.2f}, "
            f"prd={self.prd:.3f}, vei_conclusion={self.vei_conclusion!r})"
        )

    def __str__(self) -> str:
        """The report as an aligned text table (ASCII only, so it prints anywhere)."""
        rows = _report_rows(self)
        header = ("Statistic", "Value", f"{_percent(self.confidence_level)} CI", "Result")
        labels = [f"  {_ascii(row.statistic)}" for row in rows]
        widths = [
            max(len(text) for text in [header[0], *labels, *(row.section for row in rows)]),
            max(len(text) for text in [header[1], *(row.value_text for row in rows)]),
            max(len(text) for text in [header[2], *(row.interval_text for row in rows)]),
            max(len(text) for text in [header[3], *(_ascii(row.result) for row in rows)]),
        ]

        def line(statistic: str, value: str, interval: str, result: str) -> str:
            return f"{statistic:<{widths[0]}}  {value:>{widths[1]}}  {interval:>{widths[2]}}  {result}".rstrip()

        lines = [f"Ratio Study Report: {self._heading()}"]
        if self.cod_standard is not None:
            lines.append(f"COD standard: {_cod_standard_text(self.cod_standard)}")
        lines += ["", line(*header), line(*("-" * width for width in widths))]
        section = None
        for label, row in zip(labels, rows):
            if row.section != section:
                section = row.section
                lines.append(section)
            lines.append(line(label, row.value_text, row.interval_text, _ascii(row.result)))
        lines += ["", *textwrap.wrap(_ascii(_REPORT_SOURCE), width=sum(widths) + 6)]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """What notebooks (Jupyter, VS Code, Colab) show: the markdown table."""
        return self.to_markdown()

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """What IPython consoles (the terminal, Spyder) show: the text table."""
        printer.text(str(self))

    def _heading(self) -> str:
        """'54 sales, 90% confidence intervals'."""
        sales = f"{self.n_sales} sale" + ("" if self.n_sales == 1 else "s")
        return f"{sales}, {_percent(self.confidence_level)} confidence intervals"

    def to_markdown(self) -> str:
        """
        The report as a markdown table, for documents, GitHub and notebooks.

        The same table a notebook shows: Table 10.2's statistics (IAAO
        Standard on Ratio Studies 2026) and the measures of Section 8.1.3,
        rounded as Table 10.2 rounds them. Every dollar sign is escaped
        (``\\$``), because notebooks and markdown viewers read the text
        between two dollar signs as a formula.

        Returns
        -------
        str
            The markdown: a heading line, the table, and a line naming the
            sources.

        Examples
        --------
        >>> report = rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"])
        >>> with open("ratio_study.md", "w", encoding="utf-8") as file:
        ...     file.write(report.to_markdown())
        """
        heading = f"**Ratio Study Report**: {self._heading()}."
        if self.cod_standard is not None:
            heading += f" COD standard: {_cod_standard_text(self.cod_standard)}."
        lines = [
            _markdown_text(heading),
            "",
            _markdown_row(["Statistic", "Value", f"{_percent(self.confidence_level)} CI", "Result"]),
            "|:--|--:|--:|:--|",
        ]
        section = None
        for row in _report_rows(self):
            if row.section != section:
                section = row.section
                lines.append(_markdown_row([f"**{section}**", "", "", ""]))
            lines.append(_markdown_row([row.statistic, row.value_text, row.interval_text, row.result]))
        lines += ["", f"*{_markdown_text(_REPORT_SOURCE)}*"]
        return "\n".join(lines) + "\n"

    def to_frame(self) -> pd.DataFrame:
        """
        The report as a pandas DataFrame, with the unrounded values.

        One row per statistic of the table (Table 10.2 of the IAAO Standard
        on Ratio Studies 2026, and Section 8.1.3's measures), plus the PRB's
        t-value on its own row, so it can be saved to Excel or CSV or
        combined with other studies.

        Returns
        -------
        pandas.DataFrame
            Columns ``Section``, ``Statistic``, ``Value``, ``CI_Lower``,
            ``CI_Upper`` (NaN when there is no interval) and ``Result``. A
            value that can't be calculated is NaN. Percentages (the COD,
            MAPE, VEI, ...) are in percent; the test of normality's value is
            its p-value.

        Examples
        --------
        >>> report = rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"])
        >>> report.to_frame().to_excel("ratio_study.xlsx", index=False)
        """
        records = []
        for row in _report_rows(self):
            if row.statistic == _PRB_LABEL:
                records.append((row.section, "PRB", row.value, row.lower, row.upper, row.result))
                records.append((row.section, "PRB t-value", _as_number(self.prb_t_value), math.nan, math.nan, ""))
            else:
                records.append((row.section, row.statistic, row.value, row.lower, row.upper, row.result))
        return pd.DataFrame(records, columns=["Section", "Statistic", "Value", "CI_Lower", "CI_Upper", "Result"])

    def to_dict(self) -> dict:
        """
        Every attribute of the report, as a dictionary.

        The statistics of Table 10.2 of the IAAO Standard on Ratio Studies
        (2026) and of Section 8.1.3, unrounded, under their attribute names.

        Returns
        -------
        dict
            Attribute name: value, in the order of the attributes.

        Examples
        --------
        >>> report = rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"])
        >>> report.to_dict()["cod"]
        """
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _ascii(text: str) -> str:
    """The report's text for print(): '+-' for the plus-minus sign."""
    return text.replace("±", "+-")


def _markdown_text(text: str) -> str:
    """Text for markdown: a dollar sign escaped, so it is not read as the start of a formula."""
    return text.replace("$", "\\$")


def _markdown_row(cells: List[str]) -> str:
    """A markdown table row; an empty cell is a single space. Dollar signs and pipes are escaped."""
    escaped = [_markdown_text(cell).replace("|", "\\|") for cell in cells]
    return "|" + "|".join(f" {cell} " if cell else " " for cell in escaped) + "|"


def _as_number(value: Optional[float]) -> float:
    """A value for to_frame(): NaN for None."""
    return math.nan if value is None else float(value)


def _is_missing(value: Optional[float]) -> bool:
    return value is None or math.isnan(value)


def _number_text(value: Optional[float], decimals: int, suffix: str = "") -> str:
    return _NOT_CALCULATED if _is_missing(value) else f"{value:.{decimals}f}{suffix}"


def _range_text(lower: Optional[float], upper: Optional[float], decimals: int, suffix: str = "") -> str:
    if _is_missing(lower) or _is_missing(upper):
        return _NOT_CALCULATED
    return f"{lower:.{decimals}f}{suffix} to {upper:.{decimals}f}{suffix}"


def _cod_standard_text(standard: _CodStandard) -> str:
    """'Table 11.1, residential improved (5.0 to 15.0)', or 'your local standard (5.0 to 12.0)'."""
    where = f"{standard.source}, {standard.label}" if standard.source else standard.label
    return f"{where} ({_cod_limits_text(standard)})"


def _report_cod_result(report: RatioStudyReport) -> str:
    """The COD's verdict and where the COD lies against the range (Sections 11.2 and 11.2.5)."""
    standard = report.cod_standard
    if standard is None:
        return ""
    if report.cod_conclusion is None:
        return "Not calculated (too few sales)"
    low, high = _cod_limit_text(standard.low), _cod_limit_text(standard.high)
    if report.cod_conclusion == 'FAIL':
        return f"FAIL: CI entirely above {high}"
    cod = round(report.cod, 10)  # a COD exactly at a limit counts as within, as in the summary's note
    if cod > standard.high:
        return f"PASS: the COD is above {high}, but its CI is not entirely above"
    if cod < standard.low:
        return f"PASS: below {low}, a diagnostic warning (Section 11.2.5)"
    return f"PASS: within {low} to {high}"


def _report_prb_result(report: RatioStudyReport) -> str:
    """The PRB's verdict: its whole interval against +-0.05 (``PRB_Conclusion``)."""
    if report.prb_conclusion is None:
        return "Not calculated (too few sales)"
    if report.prb_conclusion == 'FAIL':
        side = "above 0.05" if round(report.prb_lower, 10) > _PRB_LIMIT else "below -0.05"
        return f"FAIL: CI entirely {side}"
    return "PASS"


def _report_prd_result(prd: float) -> str:
    """The Standard's range for the PRD, 0.98 to 1.03 (Section 11.2.6)."""
    if math.isnan(prd):
        return ""
    if prd < 0.98:
        return "Below 0.98: abnormally low"
    if prd > 1.03:
        return "Above 1.03: abnormally high"
    return "Within 0.98 to 1.03"


def _report_rows(report: RatioStudyReport) -> List[_ReportRow]:
    """The report's statistics in Table 10.2's order, then Section 8.1.3's, each with its cells."""
    rows: List[_ReportRow] = []

    def add(section, statistic, value, value_text, lower=None, upper=None, interval_text="", result=""):
        rows.append(_ReportRow(section, statistic, _as_number(value), _as_number(lower), _as_number(upper),
                               value_text, interval_text, result))

    def level(statistic, value, lower, upper):
        add("Level", statistic, value, _number_text(value, 4), lower, upper, _range_text(lower, upper, 4))

    r = report
    add("Sales", "Number of sales", r.n_sales, str(r.n_sales))
    if r.n_left_out:
        add("Sales", "Sales left out (unusable data)", r.n_left_out, str(r.n_left_out))
    add("Sales", "Total appraised value", r.total_appraised_value, f"${r.total_appraised_value:,.0f}")
    add("Sales", "Total sale price", r.total_sale_price, f"${r.total_sale_price:,.0f}")
    add("Sales", "Average appraised value", r.average_appraised_value, f"${r.average_appraised_value:,.0f}")
    add("Sales", "Average sale price", r.average_sale_price, f"${r.average_sale_price:,.0f}")

    level("Median ratio", r.median_ratio, r.median_ratio_lower, r.median_ratio_upper)
    level("Mean ratio", r.mean_ratio, r.mean_ratio_lower, r.mean_ratio_upper)
    level("Weighted mean ratio", r.weighted_mean_ratio, r.weighted_mean_lower, r.weighted_mean_upper)

    add("Uniformity", "COD", r.cod, _number_text(r.cod, 2, "%"), r.cod_lower, r.cod_upper,
        _range_text(r.cod_lower, r.cod_upper, 3, "%"), _report_cod_result(r))
    add("Uniformity", "Within ±15% of the median ratio (coefficient of concentration)",
        r.coefficient_of_concentration, _number_text(r.coefficient_of_concentration, 1, "%"))

    add("Accuracy", "MAPE", r.mape, _number_text(r.mape, 2, "%"))
    add("Accuracy", "MdAPE", r.mdape, _number_text(r.mdape, 2, "%"))
    add("Accuracy", "PE10 (within ±10% of the sale price)", r.pe10, _number_text(r.pe10, 1, "%"))
    add("Accuracy", "PE20 (within ±20% of the sale price)", r.pe20, _number_text(r.pe20, 1, "%"))

    add("Vertical equity", "VEI", r.vei, _number_text(r.vei, 2, "%"), r.vei_lower, r.vei_upper,
        _range_text(r.vei_lower, r.vei_upper, 2, "%"), r.vei_conclusion)
    prb_text = _number_text(r.prb, 4)
    if not _is_missing(r.prb_t_value):
        prb_text += f" ({r.prb_t_value:.3f})"
    add("Vertical equity", _PRB_LABEL, r.prb, prb_text, r.prb_lower, r.prb_upper,
        _range_text(r.prb_lower, r.prb_upper, 4), _report_prb_result(r))
    add("Vertical equity", "PRD", r.prd, _number_text(r.prd, 3), result=_report_prd_result(r.prd))

    p_value = _NOT_CALCULATED if _is_missing(r.normality_p_value) else f"p {_format_p_value(r.normality_p_value)}"
    add("Distribution",
        f"Normal distribution of ratios ({r.normality_method}, {r.normality_significance_level:g})",
        r.normality_p_value, p_value, result=r.normality_conclusion)
    return rows


def Ratio_Study_Report(
    y: np.ndarray,
    x: np.ndarray,
    property_type: Union[str, Tuple[float, float], None] = "none",
    confidence_level: float = 0.90,
) -> RatioStudyReport:
    """
    Ratio study report: every statistic of a ratio study, in one table.

    Calculates the statistics of the IAAO Standard on Ratio Studies (2026)
    demonstration ratio study report (Table 10.2), in its order: the number
    of sales, their totals and averages; the three measures of the level of
    valuation (the median, mean and weighted mean ratios) and the COD, each
    with its confidence interval; the VEI, PRB and PRD; and whether the
    ratios are normally distributed. It adds the measures of Section 8.1.3
    (the coefficient of concentration, MAPE and MdAPE) and PE10 / PE20. With
    a ``property_type``, the COD is also tested against the Standard's range
    (``COD_Conclusion``).

    It returns a ``RatioStudyReport``, which shows itself as a table: in a
    notebook, just evaluate it; elsewhere, ``print`` it. It prints nothing
    by itself. Each statistic is also an attribute (``report.cod``), at full
    precision, and every one of them has its own function in this module.

    .. versionadded:: 0.3.0
       Replaces ``DOR_SUMMARY_Statistics``, which will be removed in
       pyRealEstate 0.4.0. The report holds every value of its dictionary,
       under the same names in lower case (``'COD_Lower'`` is
       ``report.cod_lower``; ``'Weighted_Mean'`` is
       ``report.weighted_mean_ratio``).

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    property_type : str or (float, float), default "none"
        The COD standard to test the COD against. By default (``"none"``, or
        None) there is no COD verdict, so no property type is assumed. Name
        the property type and market of the study to compare the COD with
        the Standard's range (Tables 11.1 and 11.2) and test it against the
        range's maximum, or give your own ``(low, high)`` range for a local
        standard. Names may use any capitalization, and spaces, hyphens or
        underscores ("Vacant Land" works). Every accepted name, with its COD
        range and the Standard's row:

        Table 11.1, ratio study uniformity standards::

            "residential"               5.0 to 15.0   Residential improved
            "residential_homogeneous"   4.0 to 10.0   - newer or more homogeneous
                                                        market areas such as condominiums
            "residential_rural"        10.0 to 20.0   - rural, seasonally occupied
                                                        housing, or manufactured homes
            "multi_unit"                7.0 to 20.0   Multi-unit apartments
            "multi_unit_rural"         10.0 to 25.0   - rural jurisdictions
            "commercial"               10.0 to 25.0   Commercial, industrial, retail,
                                                      warehouse, etc.
            "commercial_condominium"    5.0 to 15.0   - commercial/industrial condominiums
            "commercial_less_active"   10.0 to 30.0   - less active markets with small
                                                        sample size
            "vacant_land"              10.0 to 20.0   Vacant land
            "vacant_land_less_active"  10.0 to 25.0   - less active markets with small
                                                        sample size
            "vacant_land_rural"        10.0 to 30.0   - rural jurisdictions
            "agricultural"             10.0 to 25.0   Agricultural land

        Table 11.2, blighted, distressed, and emerging market areas::

            "residential_distressed"   10.0 to 25.0   Residential improved, distressed
                                                      or blighted market areas
            "residential_emerging"     10.0 to 35.0   Residential improved, emerging
            "multi_unit_distressed"    10.0 to 30.0   Multi-family, distressed or blighted
            "multi_unit_emerging"      10.0 to 35.0   Multi-family, emerging
            "commercial_distressed"    10.0 to 35.0   Commercial, industrial, ...,
                                                      distressed or blighted
            "commercial_emerging"      10.0 to 40.0   Commercial, industrial, ..., emerging
            "vacant_land_distressed"   10.0 to 30.0   Vacant land, distressed or blighted
            "vacant_land_emerging"     10.0 to 45.0   Vacant land, emerging
            "agricultural_distressed"  10.0 to 45.0   Agricultural land (the table has one
            "agricultural_emerging"    10.0 to 45.0   row for distressed, blighted and
                                                      emerging market areas)

        And::

            "none"                     (the default) no COD verdict
            (low, high)                your local standard, e.g. (5.0, 12.0)

    confidence_level : float, default 0.90
        The two-tailed confidence level of every interval, between 0 and 1.
        The Standard uses 90% (Section 11.1.2, Table 10.2).

    Returns
    -------
    RatioStudyReport
        The statistics, as attributes, and the table: ``print(report)``,
        ``report.to_markdown()``, ``report.to_frame()`` (a DataFrame, for
        Excel or CSV) and ``report.to_dict()``. A statistic that can't be
        calculated is NaN (None for the PRB with 2 or fewer sales) and
        shows as "not calculated".

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``property_type`` is not an accepted name or a range of two numbers
        with 0 <= low < high; or if ``confidence_level`` is not between 0 and
        1. The options are checked first, so nothing is calculated when one
        is wrong.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation), if there are too few sales for
        the VEI (it needs 20) or a confidence interval, or if the test of
        normality can't be done or is approximate. Each statistic checks the
        data, but each problem is reported once.

    See Also
    --------
    Median_Ratio, Mean_Ratio, Weighted_Mean_Ratio, COD, PRD, PRB, VEI,
    Normality_Test, MAPE, MdAPE, Coefficient_of_Concentration, PE : each
        statistic on its own, with its formula and options.
    COD_Conclusion : the COD's verdict, with the Appendix D.8 tolerance test
        as another method.

    Notes
    -----
    Each statistic is calculated as its own function calculates it (see
    their documentation for the formulas), with the same result. The PRB's
    regression is fitted once for its coefficient, t-value, interval and
    verdict.

    **Rounding.** The table rounds as Table 10.2 does: ratios to 4 decimals,
    the COD to 2 (its interval to 3), the VEI to 2, the PRB to 4 and its
    t-value to 3, the PRD to 3, dollars to whole dollars; the MAPE and MdAPE
    to 2 decimals, the percentages of sales to 1. For the Standard's 54
    example sales (Table 10.1) it shows every value of Table 10.2 except
    two, which differ by one unit in the last digit because the Standard
    rounded differently: the average sale price, $537,418.52, shows as
    $537,419 (Table 10.2: $537,418), and the PRB's lower limit, 0.086457, as
    0.0865 (Table 10.2: 0.0864, from a t-value rounded to 3 decimals).

    **The results:**

    * **COD**, with a ``property_type``: 'FAIL' when its confidence interval
      lies entirely above the range's maximum (``COD_Conclusion``'s default
      test, one-sided at (1 + ``confidence_level``) / 2, which is the
      interval shown), otherwise 'PASS', with where the COD lies: within the
      range, above it (but not significantly), or below it, which the
      Standard says "should be treated as a diagnostic warning", for example
      of sales chasing (Section 11.2.5). The Standard's ranges "are provided
      for guidance only and may not represent jurisdictional requirements"
      (Section 11.2).
    * **VEI**: ``VEI_Conclusion``'s verdict (Appendix E.7).
    * **PRB**: 'FAIL' when its whole confidence interval lies outside
      +-0.05 (``PRB_Conclusion``), otherwise 'PASS'.
    * **PRD**: outside 0.98 to 1.03 is noted as abnormally low or high
      (Section 11.2.6). The Standard says this is the last edition in which
      the PRD is an acceptable measure (footnote 2 of Section 11).
    * **Normality**: ``Normality_Test``'s conclusion with its defaults (the
      Shapiro-Wilk test at 0.05), and its p-value. Table 10.2 says "Reject"
      for the 54 example sales without naming its test; none of the three
      tests of Table 6.3 gives that result. When the ratios are not
      normally distributed, the mean-based statistics are less reliable than
      the median and the COD (Section 6.2).
    * **The level** has no verdict: the Standard's range, 0.90 to 1.10
      (Section 11.1), is measured against the legally required level, which
      the report cannot know. Compare the intervals with your legal level.

    **Accuracy.** The MAPE, MdAPE, PE10 and PE20 measure each appraised value
    against its sale price, so they combine the level and the uniformity:
    at a median ratio of 0.87, even perfectly uniform appraisals have a MAPE
    of 13%. The coefficient of concentration and the COD measure uniformity
    around the median.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> report = rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"],
    ...                                 property_type="residential")
    >>> report                   # in a notebook: the table
    >>> print(report)            # anywhere: the table as text
    >>> report.cod, report.cod_conclusion, report.mape, report.pe10
    >>> report.to_frame().to_excel("ratio_study.xlsx", index=False)
    >>> rem.Ratio_Study_Report(sales["Sale_Price"], sales["Appraised_Value"],
    ...                        property_type=(5.0, 12.0))   # a local COD standard
    """
    cod_standard = _cod_standard(property_type)  # the options are checked first: nothing is calculated if one is wrong
    _check_fraction(confidence_level, "confidence_level", "for example 0.90 for 90%")

    # Calculate each statistic once. Each checks the data itself, so collect their data-quality
    # warnings and report each distinct problem once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataQualityWarning)
        sale_price, appraised_value, _, kept = _sales_pairs(y, x, return_rows=True)
        median_ratio = Median_Ratio(y, x)
        mean_ratio = Mean_Ratio(y, x)
        weighted_mean_ratio = Weighted_Mean_Ratio(y, x)
        cod = COD(y, x)
        # Each interval's lower and upper limit from one calculation (the helper its _Lower and _Upper
        # functions share)
        median_lower, median_upper = _median_ratio_interval(y, x, confidence_level)
        mean_lower, mean_upper = _mean_ratio_interval(y, x, confidence_level)
        weighted_mean_lower, weighted_mean_upper = _weighted_mean_interval(y, x, confidence_level)
        cod_lower, cod_upper = _cod_interval(y, x, confidence_level)
        concentration = Coefficient_of_Concentration(y, x)
        mape, mdape = MAPE(y, x), MdAPE(y, x)
        pe10, pe20 = PE(y, x), PE(y, x, within=20)
        prd = PRD(y, x)
        regression = _prb_regression(sale_price, appraised_value)  # as PRB fits it, once for all its values
        vei = VEI(y, x, confidence_level=confidence_level)
        normality = Normality_Test(y, x)  # its defaults: Shapiro-Wilk at 0.05
    _reissue_once(caught)

    if regression is None:  # 2 or fewer sales
        prb = prb_t_value = prb_lower = prb_upper = prb_conclusion = None
    else:
        prb, prb_t_value = float(regression.params[1]), float(regression.tvalues[1])
        prb_lower, prb_upper = (float(limit) for limit in _prb_interval(regression, confidence_level))
        prb_conclusion = _prb_verdict(prb_lower, prb_upper)

    return RatioStudyReport(
        n_sales=len(sale_price),
        n_left_out=int((~kept).sum()),
        total_appraised_value=float(np.sum(appraised_value)),
        total_sale_price=float(np.sum(sale_price)),
        average_appraised_value=float(np.mean(appraised_value)),
        average_sale_price=float(np.mean(sale_price)),
        median_ratio=float(median_ratio),
        median_ratio_lower=float(median_lower),
        median_ratio_upper=float(median_upper),
        mean_ratio=float(mean_ratio),
        mean_ratio_lower=float(mean_lower),
        mean_ratio_upper=float(mean_upper),
        weighted_mean_ratio=float(weighted_mean_ratio),
        weighted_mean_lower=float(weighted_mean_lower),
        weighted_mean_upper=float(weighted_mean_upper),
        cod=float(cod),
        cod_lower=float(cod_lower),
        cod_upper=float(cod_upper),
        # The interval test at (1 + level) / 2 one-sided, whose lower limit is the interval's (as COD_Conclusion)
        cod_conclusion=None if cod_standard is None else _cod_verdict(cod_lower, cod_standard.high),
        cod_standard=cod_standard,
        coefficient_of_concentration=concentration,
        mape=mape,
        mdape=mdape,
        pe10=pe10,
        pe20=pe20,
        vei=float(vei.vei),
        vei_lower=float(vei.ci_lower),
        vei_upper=float(vei.ci_upper),
        vei_conclusion=vei.conclusion,
        prb=prb,
        prb_t_value=prb_t_value,
        prb_lower=prb_lower,
        prb_upper=prb_upper,
        prb_conclusion=prb_conclusion,
        prd=float(prd),
        normality_method=normality.method,
        normality_significance_level=normality.significance_level,
        normality_p_value=float(normality.p_value),
        normality_conclusion=normality.conclusion,
        confidence_level=confidence_level,
    )


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
    level? (IAAO Standard on Ratio Studies 2026, Section 6.4, Table 6.3, and
    Appendix F, Table F.1.1.)

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
          the ratios (1 = the lowest ratio);
        * ``COD``: the group's coefficient of dispersion, in percent
          (Section 8.1.1), so the groups' uniformity can be compared too.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Kruskal_Wallis(
    ...     sales["Sale_Price"], sales["Appraised_Value"], sales["Neighborhood"]
    ... )
    >>> result.p_value, result.conclusion
    >>> print(result)                  # plain-English report
    >>> result.groups                  # each group's sales, median ratio, average rank, COD
    >>> rem.VEI(sales["Sale_Price"], sales["Appraised_Value"],
    ...         kruskal_wallis=True).kruskal_wallis   # the same test on the VEI's value groups
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

        group_table = self.groups.to_string(index=False, formatters=_RANKED_GROUP_FORMATS)
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


# How the groups table of Kruskal_Wallis and Mann_Whitney is printed
_RANKED_GROUP_FORMATS = {"Median_Ratio": "{:.4f}".format, "Mean_Rank": "{:.2f}".format, "COD": "{:.2f}".format}


def _group_labels(y: Any, groups: Any) -> np.ndarray:
    """The group labels, paired with the sales by position (categoricals become plain labels)."""
    group_labels = pd.Series(groups).to_numpy()
    if len(group_labels) != len(y):
        raise ValueError(
            f"There are {len(y)} sale prices (y) but {len(group_labels)} group labels; each sale "
            "needs exactly one group label."
        )
    return group_labels


def _ratios_with_groups(
    y: Any,
    x: Any,
    group_labels: np.ndarray,
    test_name: str,
    label_noun: str = "group",
    problem: str = "missing group label",
    return_rows: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    The sale ratios and their group labels, after the data checks (see "Data
    checks" in the module documentation). A sale without a group label cannot
    be compared with anything, so it is left out, with a warning (worded with
    ``label_noun`` and ``problem``: ``Sales_Chasing``'s split-sample test uses
    it for sale dates). With ``return_rows``, also the row labels of the sales
    kept.
    """
    sale_price, appraised_value, row_labels, kept = _sales_pairs(y, x, return_rows=True)
    group_labels = group_labels[kept]

    no_group = np.asarray(pd.isna(group_labels), dtype=bool)
    if no_group.any():
        _warn_data_quality(
            f"The {test_name} test left out {int(no_group.sum())} of {len(no_group)} sales that "
            f"have no {label_noun}:\n"
            + _problem_lines([(problem, no_group)], row_labels)
            + f"\nThe test uses the remaining {int((~no_group).sum())} sales."
        )
        has_group = ~no_group
        sale_price, appraised_value = sale_price[has_group], appraised_value[has_group]
        group_labels, row_labels = group_labels[has_group], row_labels[has_group]

    if return_rows:
        return appraised_value / sale_price, group_labels, row_labels
    return appraised_value / sale_price, group_labels


def _ranked_groups(ratio: np.ndarray, group_labels: np.ndarray) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Rank every ratio, 1 for the lowest (tied ratios share the average of
    their ranks), and describe each group: its sales, median ratio, average
    rank and COD. Returns (ratios, ranks, groups table).
    """
    order = _group_order(group_labels)
    ratios = pd.Series(ratio, dtype=float)
    ranks = ratios.rank(method="average")
    by_group = pd.DataFrame({"Ratio": ratios, "Rank": ranks}).groupby(group_labels, sort=False)
    median_ratio = by_group["Ratio"].median()

    # COD = 100 * average absolute difference from the group's median ratio / that median (Section 8.1.1)
    absolute_difference = (ratios - by_group["Ratio"].transform("median")).abs()
    cod = 100 * absolute_difference.groupby(group_labels, sort=False).mean() / median_ratio

    groups = pd.DataFrame({
        "Group": order,
        "Sales": by_group.size().reindex(order).to_numpy(),
        "Median_Ratio": median_ratio.reindex(order).to_numpy(),
        "Mean_Rank": by_group["Rank"].mean().reindex(order).to_numpy(),
        "COD": cod.reindex(order).to_numpy(),
    })
    return ratios, ranks, groups


def _kruskal_wallis_calculation(
    ratio: np.ndarray, group_labels: np.ndarray, significance_level: float
) -> KruskalWallisResult:
    """The Kruskal-Wallis test on sales that have passed the data checks."""
    n_sales = len(ratio)
    ratios, _, groups = _ranked_groups(ratio, group_labels)
    n_groups = len(groups)

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
    equity study. For two groups, use ``Mann_Whitney``: with two groups H
    is the square of the Mann-Whitney test's z, but the p-value here has no
    continuity correction and is never exact, so the two p-values differ.

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
        group's sales, median ratio, average rank and COD (``groups``).
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
    Mann_Whitney : the test for two groups.
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

    The test compares where each group's ratios sit in the ranking, which
    depends on their spread as well as their level; the ``COD`` column shows
    each group's uniformity (see ``Mann_Whitney``'s Notes).

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
    >>> result.groups                     # each group's sales, median ratio, average rank, COD
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    group_labels = _group_labels(y, groups)
    ratio, group_labels = _ratios_with_groups(y, x, group_labels, "Kruskal-Wallis")
    return _kruskal_wallis_calculation(ratio, group_labels, significance_level)


# ---------------------------------------------------------------------------
# Mann-Whitney test
#
# Source: IAAO Standard on Ratio Studies (2026): Section 6.4 (Table 6.3, "two
# property groups are appraised at equal percentages of market value"), 6.5,
# A.8, and Appendix F (Table F.1.1). The Standard names the test but gives no
# formula or worked example; the calculation is the textbook one (Mann and
# Whitney 1947), with the exact p-value for small samples.
# ---------------------------------------------------------------------------

_MW_EXACT_MAXIMUM_SALES = 50  # exact p-value up to 50 sales (user decision, 2026-11-01); see _mann_whitney_exact_p_value
_MW_SMALL_GROUP = 5  # groups with fewer sales get a warning, as in Kruskal_Wallis (user decision, 2026-11-01)

# How the p-value was calculated (MannWhitneyResult.p_value_method)
_MW_EXACT = "exact"
_MW_NORMAL = "normal approximation"
_MW_NOT_CALCULATED = "not calculated"


@dataclass(frozen=True, eq=False, repr=False)
class MannWhitneyResult:
    """
    The result of ``Mann_Whitney``: are the two groups appraised at the same
    level? (IAAO Standard on Ratio Studies 2026, Section 6.4, Table 6.3, and
    Appendix F, Table F.1.1.)

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    statistic : float
        U for the first group in ``groups``: the number of pairs (one sale
        from each group) in which the first group's ratio is the higher one,
        a tie counting half. It runs from 0 to n1 x n2, and n1 x n2 / 2 is what
        to expect when the groups do not differ. NaN with fewer than 2 groups.
    p_value : float
        The chance of a U at least this far from n1 x n2 / 2, in either
        direction, if both groups were appraised at the same level. NaN when
        it cannot be calculated: fewer than 2 groups, or every ratio identical.
    p_value_method : str
        ``'exact'`` (50 sales or fewer), ``'normal approximation'`` (more
        than 50 sales), or ``'not calculated'``.
    significance_level : float
        The significance level used for the conclusion, e.g. 0.05.
    conclusion : str
        ``'Significant difference between groups'`` (the p-value is below the
        significance level), ``'No significant difference between groups'``,
        or ``'Not enough groups (fewer than 2)'``.
    share_higher : float
        U / (n1 x n2): the share of the pairs in which the first group's ratio
        is the higher one. 0.5 means neither group tends to be appraised
        higher. NaN with fewer than 2 groups.
    groups : pandas.DataFrame
        One row per group, in sorted order of the labels (or in order of
        first appearance when the labels cannot be sorted):

        * ``Group``: the group label;
        * ``Sales``: the number of sales in the group;
        * ``Median_Ratio``: the group's median sale ratio;
        * ``Mean_Rank``: the average rank of the group's ratios among all
          the ratios (1 = the lowest ratio);
        * ``COD``: the group's coefficient of dispersion, in percent
          (Section 8.1.1).

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Mann_Whitney(
    ...     sales["Sale_Price"], sales["Appraised_Value"], sales["Sold_or_Unsold"]
    ... )
    >>> result.p_value, result.conclusion
    >>> result.share_higher            # share of pairs where the first group's ratio is higher
    >>> print(result)                  # plain-English report
    >>> result.groups                  # each group's sales, median ratio, average rank, COD
    """

    statistic: float
    p_value: float
    p_value_method: str
    significance_level: float
    conclusion: str
    share_higher: float
    groups: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"MannWhitneyResult(statistic={self.statistic:.1f}, p_value={self.p_value:.4f}, "
            f"p_value_method={self.p_value_method!r}, conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        n_groups = len(self.groups)
        lines = [
            "Mann-Whitney test: are the two groups appraised at the same level?",
            "(IAAO Standard on Ratio Studies, 2026: Section 6.4 and Appendix F)",
            f"{int(self.groups['Sales'].sum())} sales in {n_groups} group{'' if n_groups == 1 else 's'}",
        ]
        if self.conclusion == _KW_NOT_ENOUGH_GROUPS:
            lines.append(f"Conclusion: {self.conclusion}")
            return "\n".join(lines)

        lines += [
            self._statistic_text(),
            f"Conclusion at the {_percent(self.significance_level)} significance level: "
            f"{self.conclusion}",
            f"In {self.share_higher * 100:.1f}% of the pairs (one sale from each group, a tie counting "
            f"half), group {self.groups['Group'].iloc[0]} has the higher ratio.",
            "",
            self.groups.to_string(index=False, formatters=_RANKED_GROUP_FORMATS),
        ]
        return "\n".join(lines)

    def _statistic_text(self) -> str:
        """'U = 2.0, p-value = 0.0635 (exact)', or why there is no p-value."""
        if math.isnan(self.p_value):
            return (
                f"U = {self.statistic:.1f}; the p-value cannot be calculated: every sale ratio is the "
                "same, so there is nothing to rank."
            )
        return f"U = {self.statistic:.1f}, p-value {_format_p_value(self.p_value)} ({self.p_value_method})"


def _mann_whitney_exact_p_value(ranks: np.ndarray, in_first: np.ndarray) -> Tuple[float, float]:
    """
    The exact two-sided p-value, and the smallest p-value these ratios could
    give however they were split into groups of these sizes.

    If the groups do not differ, every way to choose which n of the N sales
    form a group is equally likely. The p-value is the share of those
    C(N, n) ways whose rank sum is at least as far from its expected value,
    n (N + 1) / 2, as the observed one. The ways are counted by rank sum, one
    sale at a time. Ranks are doubled, so tied (half) ranks become whole
    numbers. Every count is a whole number of at most C(50, 25) ~ 1.3e14,
    below 2**53, so floating point holds it exactly up to 50 sales.
    """
    n_sales = len(ranks)
    counted = in_first if 2 * in_first.sum() <= n_sales else ~in_first  # the smaller group: less work, same p
    n_counted = int(counted.sum())
    doubled = np.rint(2 * ranks).astype(np.int64)
    largest_sum = int(np.sort(doubled)[n_sales - n_counted:].sum())

    # ways[k, s]: the number of ways to choose k of the sales seen so far with doubled rank sum s
    ways = np.zeros((n_counted + 1, largest_sum + 1))
    ways[0, 0] = 1.0
    for seen, rank in enumerate(doubled, start=1):
        k = min(seen, n_counted)
        # Choosing this sale as well: each way to choose k - 1 of the earlier
        # sales, with its rank sum moved up by this sale's rank. (.copy():
        # read every earlier count before any is updated.)
        ways[1:k + 1, rank:] += ways[:k, :largest_sum + 1 - rank].copy()

    expected = n_counted * (n_sales + 1)  # the doubled expected rank sum
    distance = np.abs(np.arange(largest_sum + 1) - expected)
    observed = abs(int(doubled[counted].sum()) - expected)
    ways_with_n = ways[n_counted]
    total = math.comb(n_sales, n_counted)
    p_value = ways_with_n[distance >= observed].sum() / total
    farthest = distance[ways_with_n > 0].max()
    smallest_p_value = ways_with_n[distance >= farthest].sum() / total
    return min(float(p_value), 1.0), min(float(smallest_p_value), 1.0)


def _mann_whitney_normal_p_value(u: float, n1: int, n2: int, tie_sizes: np.ndarray) -> Tuple[float, float]:
    """
    The normal approximation to the two-sided p-value (corrected for ties,
    with the continuity correction), and the smallest p-value it can give
    for these group sizes (at U = 0 or n1 x n2).
    """
    n_sales = n1 + n2
    expected = n1 * n2 / 2
    tie_term = float((tie_sizes ** 3 - tie_sizes).sum()) / (n_sales * (n_sales - 1))
    standard_deviation = math.sqrt(n1 * n2 / 12 * ((n_sales + 1) - tie_term))

    def two_sided(distance: float) -> float:
        z = max(distance - 0.5, 0.0) / standard_deviation  # - 0.5: the continuity correction
        return min(math.erfc(z / math.sqrt(2)), 1.0)  # erfc(z / sqrt(2)) = P(|standard normal| > z)

    return two_sided(abs(u - expected)), two_sided(expected)


def _mann_whitney_test(ranks: pd.Series, in_first: np.ndarray) -> Tuple[float, float, str, float]:
    """
    The Mann-Whitney test from the ranks of every value (tied values share
    the average of their ranks) and which values are in the first group: U
    for the first group, the two-sided p-value, how it was calculated, and
    the smallest p-value these values could give. When every value is
    identical there is nothing to rank: the p-value is NaN and "not
    calculated". Shared by ``Mann_Whitney`` and ``Sales_Chasing``'s
    comparisons of sold and unsold parcels.
    """
    n1 = int(in_first.sum())
    n2 = len(ranks) - n1
    # U for the first group: its rank sum minus the smallest rank sum n1
    # values can have, n1 (n1 + 1) / 2. This equals the number of pairs (one
    # value from each group) in which the first group's value is higher, a tie
    # counting half.
    u = float(ranks[in_first].sum()) - n1 * (n1 + 1) / 2
    tie_sizes = ranks.value_counts().to_numpy(dtype=float)  # tied values share one rank
    if len(tie_sizes) == 1:
        return u, math.nan, _MW_NOT_CALCULATED, math.nan
    if n1 + n2 <= _MW_EXACT_MAXIMUM_SALES:
        p_value, smallest_p_value = _mann_whitney_exact_p_value(ranks.to_numpy(), in_first)
        return u, p_value, _MW_EXACT, smallest_p_value
    p_value, smallest_p_value = _mann_whitney_normal_p_value(u, n1, n2, tie_sizes)
    return u, p_value, _MW_NORMAL, smallest_p_value


def _warn_small_mann_whitney_groups(
    labels: Any,
    counts: Any,
    p_value_method: str,
    smallest_p_value: float,
    significance_level: float,
    noun: str = "sale",
) -> None:
    """
    One warning for groups of fewer than 5 sales (or parcels: ``noun``), and
    for group sizes that can never be significant. ``labels`` and ``counts``
    name the two groups and their sizes, in order.
    """
    sentences = []
    small = [(label, count) for label, count in zip(labels, counts) if count < _MW_SMALL_GROUP]
    if small:
        listed = ", ".join(f"{label} ({count} {noun}{'' if count == 1 else 's'})" for label, count in small)
        exact = ""
        if p_value_method == _MW_EXACT:
            exact = f" (with {_MW_EXACT_MAXIMUM_SALES} {noun}s or fewer its p-value is exact)"
        sentences.append(
            f"{len(small)} of 2 groups {'has' if len(small) == 1 else 'have'} fewer than {_MW_SMALL_GROUP} "
            f"{noun}s: {listed}. The Mann-Whitney test still runs{exact}, but groups this small can only "
            "show large differences."
        )
    if smallest_p_value >= significance_level:  # NaN (not calculated) never is
        n1, n2 = counts
        sentences.append(
            f"With {n1} and {n2} {noun}s, the p-value cannot be below {smallest_p_value:.4f}, so this test "
            f"cannot find a significant difference at the {_percent(significance_level)} significance "
            "level, however different the groups are."
        )
    if sentences:
        _warn_data_quality(" ".join(sentences))


def _mann_whitney_calculation(
    ratio: np.ndarray, group_labels: np.ndarray, significance_level: float
) -> MannWhitneyResult:
    """The Mann-Whitney test on sales that have passed the data checks (at most 2 groups)."""
    ratios, ranks, groups = _ranked_groups(ratio, group_labels)
    if len(groups) < 2:
        _warn_data_quality(
            "The Mann-Whitney test needs 2 groups to compare, but the usable sales are in "
            f"{len(groups)} group{'' if len(groups) == 1 else 's'}. Its statistic and p-value are left "
            "empty (NaN)."
        )
        return MannWhitneyResult(
            statistic=math.nan, p_value=math.nan, p_value_method=_MW_NOT_CALCULATED,
            significance_level=significance_level, conclusion=_KW_NOT_ENOUGH_GROUPS,
            share_higher=math.nan, groups=groups,
        )

    in_first = np.asarray(group_labels == groups["Group"].iloc[0], dtype=bool)
    n1, n2 = (int(n) for n in groups["Sales"])
    u, p_value, p_value_method, smallest_p_value = _mann_whitney_test(ranks, in_first)
    share_higher = u / (n1 * n2)

    def result(p_value: float, p_value_method: str, conclusion: str) -> MannWhitneyResult:
        return MannWhitneyResult(
            statistic=u, p_value=p_value, p_value_method=p_value_method,
            significance_level=significance_level, conclusion=conclusion,
            share_higher=share_higher, groups=groups,
        )

    if p_value_method == _MW_NOT_CALCULATED:  # every ratio is identical: there is nothing to rank
        _warn_small_mann_whitney_groups(
            groups["Group"], groups["Sales"], _MW_NOT_CALCULATED, math.nan, significance_level
        )
        _warn_data_quality(
            f"Every sale ratio is the same ({ratios.iloc[0]:.4f}), so the Mann-Whitney test's p-value "
            "cannot be calculated: there is nothing to rank. The groups do not differ. Identical "
            "ratios are unusual in real data; please check them."
        )
        return result(math.nan, _MW_NOT_CALCULATED, _KW_NOT_SIGNIFICANT)

    _warn_small_mann_whitney_groups(
        groups["Group"], groups["Sales"], p_value_method, smallest_p_value, significance_level
    )
    conclusion = _KW_SIGNIFICANT if p_value < significance_level else _KW_NOT_SIGNIFICANT
    return result(p_value, p_value_method, conclusion)


def Mann_Whitney(
    y: np.ndarray,
    x: np.ndarray,
    groups: Any,
    significance_level: float = 0.05,
) -> MannWhitneyResult:
    """
    Mann-Whitney test: are two groups appraised at the same level?

    Tests whether the sales of two groups (two neighborhoods, two property
    classes, sales vs independent value estimates, ...) are appraised at the
    same percentage of market value. Every sale ratio is ranked from lowest
    to highest; if one group's ratios sit systematically higher in that
    ranking than the other's, the groups are not appraised at the same
    level. The test uses ranks, not the ratios themselves, so it does not
    assume the ratios follow a normal distribution, and outliers have little
    influence.

    The IAAO Standard on Ratio Studies (2026) lists it in Section 6.4,
    Table 6.3, as the test that "two property groups are appraised at equal
    percentages of market value"; in A.8, to compare ratios based on sales
    with ratios based on independent value estimates; and in Appendix F
    (Table F.1.1), where it "may be useful for smaller sample sizes". It is
    the two-group version of ``Kruskal_Wallis``.

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    groups : array-like
        The group of each sale (e.g. a neighborhood code), in the same order
        as ``y``: exactly two different labels. Any labels work: text,
        numbers, or a pandas categorical.
    significance_level : float, default 0.05
        The groups are said to differ significantly when the p-value is below
        this level, e.g. 0.05 for 95% confidence.

    Returns
    -------
    MannWhitneyResult
        U (``statistic``), its ``p_value`` and how it was calculated
        (``p_value_method``), the ``conclusion``, the share of pairs in which
        the first group's ratio is higher (``share_higher``), and a table with
        each group's sales, median ratio, average rank and COD (``groups``).
        ``print(result)`` shows a plain-English report.

    Raises
    ------
    ValueError
        If ``groups`` has more than 2 different labels (use
        ``Kruskal_Wallis``); if the inputs differ in length (including
        ``groups``), are empty, are not numbers, or have no usable sales; or
        if ``significance_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out (see "Data checks" in the module documentation),
        including sales without a group label; if a group has fewer than 5
        sales; if the group sizes make a significant result impossible; if
        fewer than 2 groups remain; or if every ratio is identical.

    See Also
    --------
    Kruskal_Wallis : the same idea for three or more groups.

    Notes
    -----
    The calculation (Mann and Whitney 1947; the Standard names the test but
    gives no formula)::

        ratio    = appraised value / sale price
        rank     = position of each ratio among all N = n1 + n2 ratios,
                   lowest = 1 (tied ratios share the average of their positions)
        U        = (sum of the first group's ranks) - n1 (n1 + 1) / 2
                 = the number of pairs, one sale from each group, in which the
                   first group's ratio is higher (a tie counts 1/2)
        share_higher = U / (n1 n2)

    If the groups do not differ, U is n1 n2 / 2 on average. The two-sided
    p-value is the chance of a U at least as far from n1 n2 / 2 as the one
    observed:

    * **Exact, with 50 sales or fewer** (ties included). Every way to split
      the N ratios into groups of n1 and n2 is equally likely if the groups
      do not differ; the p-value is the share of the C(N, n1) splits whose U
      is at least as far from n1 n2 / 2. With small groups this matters: for
      4 and 5 sales without ties, the exact p-value can be 0.0635 where the
      normal approximation gives 0.0662, or 0.0500 without the continuity
      correction.
    * **Normal approximation, above 50 sales**, corrected for ties and with
      the continuity correction::

          variance = n1 n2 / 12 * ((N + 1) - sum over runs of t tied ratios of (t^3 - t) / (N (N - 1)))
          z        = (|U - n1 n2 / 2| - 1/2) / sqrt(variance)
          p-value  = chance that a standard normal value is farther than z from 0

      Above 50 sales it stays within about 0.004 of the exact p-value
      (measured on simulated samples).

    The conclusion is "Significant difference between groups" when the
    p-value is below ``significance_level``.

    With tied ratios and 50 sales or fewer, the p-value differs from what
    scipy (``scipy.stats.mannwhitneyu``) and R (``wilcox.test``) print by
    default: they switch to the normal approximation when there are ties.
    This function counts the splits exactly, ties included. (scipy's
    ``method="exact"`` ignores ties, so its p-value is wrong when ratios tie.)

    **Not just a test of medians.** The test compares where the two groups'
    ratios sit in the ranking, which depends on their spread as well as their
    level (Hart 2001, cited by the Standard; Section 6.5 says it applies
    "given similar distribution of ratios"). Two groups with the same median
    ratio but very different uniformity can test as different. The ``COD``
    column of ``groups`` shows each group's uniformity; when the CODs differ
    a lot, read a significant result with care.

    **Small groups.** The p-value is exact with 50 sales or fewer, but small
    groups can only show large differences: a group with fewer than 5 sales
    gets a warning (the same rule of thumb as ``Kruskal_Wallis``). Some group
    sizes can never give a significant result: with 2 and 3 sales the
    p-value cannot be below 0.2, and with 1 sale against 39 or fewer others
    it is never below 0.05. The warning says so, with the smallest p-value
    possible.

    Special cases, which give NaN rather than an error so that a loop over
    many strata keeps running:

    * Fewer than 2 groups (for example, every sale of one group was left
      out): conclusion ``'Not enough groups (fewer than 2)'``.
    * Every ratio identical: every pair ties, so U = n1 n2 / 2, but there is
      nothing to rank; the p-value is NaN and the conclusion is "No
      significant difference between groups".

    The first group is the first label in sorted order (or in order of first
    appearance when the labels cannot be sorted). Swapping the groups turns U
    into n1 n2 - U and leaves the p-value unchanged. Group labels are paired
    with the sales by position, like the sale prices and appraised values.
    Sales without a group label (None, NaN) are left out, with a warning.
    Appraised values of 0 or less are kept, with the usual warning (ranks
    need no logarithm).

    Comparing sold and unsold parcels (Table 6.3; Appendix B) uses value
    changes or values per unit rather than sale ratios, because unsold
    parcels have no sale price; this function compares sale ratios.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Mann_Whitney(
    ...     sales["Sale_Price"], sales["Appraised_Value"], sales["Property_Class"]
    ... )
    >>> print(result)                     # plain-English report
    >>> result.p_value, result.conclusion
    >>> result.groups                     # each group's sales, median ratio, average rank, COD
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    group_labels = _group_labels(y, groups)

    labels = _group_order(group_labels[~np.asarray(pd.isna(group_labels), dtype=bool)])
    if len(labels) > 2:
        listed = ", ".join(str(label) for label in labels[:10])
        if len(labels) > 10:
            listed += f", and {len(labels) - 10} more"
        raise ValueError(
            f"The Mann-Whitney test compares exactly 2 groups, but `groups` has {len(labels)} "
            f"({listed}). For 3 or more groups, use Kruskal_Wallis; to compare two of them, select "
            "their sales first."
        )

    ratio, group_labels = _ratios_with_groups(y, x, group_labels, "Mann-Whitney")
    return _mann_whitney_calculation(ratio, group_labels, significance_level)


# ---------------------------------------------------------------------------
# Test of normality
#
# Source: IAAO Standard on Ratio Studies (2026): Section 6 (Principles and
# 6.2), Table 6.3 ("Ratios are normally distributed": the Shapiro-Wilk W,
# D'Agostino-Pearson K2 and Anderson-Darling A2 tests), and its References:
# D'Agostino and Stephens (1986); D'Agostino, Belanger and D'Agostino (1990).
# ---------------------------------------------------------------------------

# The conclusions returned by Normality_Test
_NORMALITY_REJECTED = "Not normally distributed"
_NORMALITY_NOT_REJECTED = "No significant departure from normality"
_NORMALITY_IDENTICAL_RATIOS = "Not calculated (every sale ratio is the same)"

# The tests of Table 6.3, by simplified name: (name, the statistic's symbol, the fewest sales the test needs)
_NORMALITY_TESTS = {
    "shapirowilk": ("Shapiro-Wilk", "W", 3),
    "dagostinopearson": ("D'Agostino-Pearson", "K2", 8),
    "andersondarling": ("Anderson-Darling", "A2", 8),
}
_NORMALITY_SHORT_NAMES = {"shapiro": "shapirowilk", "dagostino": "dagostinopearson", "anderson": "andersondarling"}
_SHAPIRO_WILK_MAXIMUM_SALES = 5000  # Royston's approximation of its p-value is fitted up to 5,000 sales


@dataclass(frozen=True, eq=False, repr=False)
class NormalityTestResult:
    """
    The result of ``Normality_Test``: are the sale ratios normally
    distributed? (IAAO Standard on Ratio Studies 2026, Section 6 and
    Table 6.3.)

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    method : str
        The test: ``'Shapiro-Wilk'``, ``"D'Agostino-Pearson"`` or
        ``'Anderson-Darling'``.
    statistic : float
        The test's statistic: W, K2 or A2. NaN when the test can't be done.
    p_value : float
        The chance of a departure from normality at least this large if the
        ratios came from a normal distribution. NaN when the test can't be
        done.
    significance_level : float
        The significance level used for the conclusion, e.g. 0.05.
    conclusion : str
        ``'Not normally distributed'`` (the p-value is below the significance
        level; Table 10.2's "Reject"), ``'No significant departure from
        normality'`` (which is not proof that the ratios are normal), or why
        the test can't be done: ``'Not enough sales (fewer than 8)'``, or
        ``'Not calculated (every sale ratio is the same)'``.
    n_sales : int
        The number of usable sales.
    skewness : float
        How lopsided the ratios are: 0 for a normal distribution, positive
        for a long tail of high ratios. NaN when every ratio is the same.
    excess_kurtosis : float
        How heavy the tails are: 0 for a normal distribution, positive for
        more extreme ratios than normal, negative for fewer. NaN when every
        ratio is the same.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Normality_Test(sales["Sale_Price"], sales["Appraised_Value"])
    >>> result.p_value, result.conclusion
    >>> result.skewness, result.excess_kurtosis
    >>> print(result)                  # plain-English report
    """

    method: str
    statistic: float
    p_value: float
    significance_level: float
    conclusion: str
    n_sales: int
    skewness: float
    excess_kurtosis: float

    def __repr__(self) -> str:
        return (
            f"NormalityTestResult(method={self.method!r}, statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        lines = [
            "Test of normality: are the sale ratios normally distributed?",
            "(IAAO Standard on Ratio Studies, 2026: Section 6 and Table 6.3)",
            f"{self.method} test, {self.n_sales} sales",
        ]
        if math.isnan(self.statistic):
            lines.append(f"Conclusion: {self.conclusion}")
        else:
            symbol = next(symbol for name, symbol, _ in _NORMALITY_TESTS.values() if name == self.method)
            lines += [
                f"{symbol} = {self.statistic:.4f}, p-value {_format_p_value(self.p_value)}",
                f"Conclusion at the {_percent(self.significance_level)} significance level: {self.conclusion}",
            ]
        if not math.isnan(self.skewness):
            lines.append(
                f"Shape of the ratios: skewness {self.skewness:.3f}, excess kurtosis {self.excess_kurtosis:.3f} "
                "(both 0 for a normal distribution)"
            )
        return "\n".join(lines)


def _check_normality_method(method: Any) -> str:
    """The simplified name of one of Table 6.3's tests ('shapirowilk', ...); anything else raises a ValueError."""
    key = _simplified_name(method).replace("'", "") if isinstance(method, str) else None
    key = _NORMALITY_SHORT_NAMES.get(key, key)
    if key not in _NORMALITY_TESTS:
        raise ValueError(
            "method must be 'shapiro-wilk', 'dagostino-pearson' or 'anderson-darling' (the tests of normality in "
            f"Table 6.3 of the Standard), not {method!r}."
        )
    return key


def _skewness_and_excess_kurtosis(ratio: np.ndarray) -> Tuple[float, float]:
    """
    skewness = m3 / m2**1.5 and excess kurtosis = m4 / m2**2 - 3, with m_k
    the average of (ratio - mean ratio)**k; (NaN, NaN) when every ratio is
    the same.
    """
    deviation = ratio - np.mean(ratio)
    m2 = np.mean(deviation**2)
    if m2 == 0:
        return math.nan, math.nan
    return float(np.mean(deviation**3) / m2**1.5), float(np.mean(deviation**4) / m2**2 - 3)


def _polynomial(coefficients: List[float], x: float) -> float:
    """coefficients[0] + coefficients[1] * x + coefficients[2] * x**2 + ..."""
    return sum(coefficient * x**power for power, coefficient in enumerate(coefficients))


def _shapiro_wilk(ratio: np.ndarray) -> Tuple[float, float]:
    """
    The Shapiro-Wilk W and its p-value by Royston's approximation (1992,
    1995; Algorithm AS R94, the algorithm scipy and R use), in double
    precision. (scipy 1.10 computes it partly in single precision, which
    moved p-values by up to 0.004 at 5,000 sales.)
    """
    from scipy import stats  # imported here: slow to import (see top of module)

    x = np.sort(ratio)
    n = len(x)
    half = n // 2

    # The coefficients a_1 .. a_half of the largest ratios (the smallest get -a_i), from the expected
    # positions of normal order statistics m_i, with Royston's polynomial corrections for the two outermost
    if n == 3:
        a = np.array([math.sqrt(0.5)])
    else:
        m = stats.norm.ppf((np.arange(1, half + 1) - 0.375) / (n + 0.25))  # negative: the lower half
        sum_m2 = 2 * np.sum(m**2)
        u = 1 / math.sqrt(n)
        a1 = _polynomial([0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056], u) - m[0] / math.sqrt(sum_m2)
        if n > 5:
            a2 = _polynomial([0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633], u) - m[1] / math.sqrt(sum_m2)
            scale = math.sqrt((sum_m2 - 2 * m[0]**2 - 2 * m[1]**2) / (1 - 2 * a1**2 - 2 * a2**2))
            a = np.concatenate([[a1, a2], -m[2:] / scale])
        else:
            scale = math.sqrt((sum_m2 - 2 * m[0]**2) / (1 - 2 * a1**2))
            a = np.concatenate([[a1], -m[1:] / scale])

    # W = (sum of a_i * (x_(n+1-i) - x_(i)))**2 / sum of (x - mean)**2
    w = float(np.sum(a * (x[::-1][:half] - x[:half])) ** 2 / np.sum((x - np.mean(x)) ** 2))

    # The p-value: exact for 3 sales; otherwise a normal approximation of a transformation of 1 - W
    if n == 3:
        return w, max(0.0, 6 / math.pi * (math.asin(math.sqrt(w)) - math.pi / 3))
    log_one_minus_w = math.log(1 - w)
    if n <= 11:
        gamma = -2.273 + 0.459 * n
        if log_one_minus_w >= gamma:
            return w, 1e-99
        transformed = -math.log(gamma - log_one_minus_w)
        mean = _polynomial([0.5440, -0.39978, 0.025054, -6.714e-4], n)
        sd = math.exp(_polynomial([1.3822, -0.77857, 0.062767, -0.0020322], n))
    else:
        transformed = log_one_minus_w
        mean = _polynomial([-1.5861, -0.31082, -0.083751, 0.0038915], math.log(n))
        sd = math.exp(_polynomial([-0.4803, -0.082676, 0.0030302], math.log(n)))
    return w, float(stats.norm.sf((transformed - mean) / sd))


def _anderson_darling(ratio: np.ndarray) -> Tuple[float, float]:
    """
    The Anderson-Darling A2 for a normal distribution with an estimated mean
    and standard deviation, and its p-value from D'Agostino and Stephens
    (1986). The formulas are in ``Normality_Test``'s Notes.
    """
    from scipy import stats  # imported here: slow to import (see top of module)

    n = len(ratio)
    z = np.sort((ratio - np.mean(ratio)) / np.std(ratio, ddof=1))
    i = np.arange(1, n + 1)
    # ln F(z_i) + ln(1 - F(z_(n+1-i))), with scipy's log functions, which stay accurate far into the tails
    log_terms = stats.norm.logcdf(z) + stats.norm.logsf(z[::-1])
    a2 = float(-n - np.sum((2 * i - 1) * log_terms) / n)

    a2_star = a2 * (1 + 0.75 / n + 2.25 / n**2)
    if a2_star >= 0.6:
        # This curve has its minimum at 5.709 / (2 * 0.0186) = 153.47 (p about 1e-190) and rises after it
        a2_star = min(a2_star, 5.709 / (2 * 0.0186))
        p_value = math.exp(1.2937 - 5.709 * a2_star + 0.0186 * a2_star**2)
    elif a2_star >= 0.34:
        p_value = math.exp(0.9177 - 4.279 * a2_star - 1.38 * a2_star**2)
    elif a2_star >= 0.2:
        p_value = 1 - math.exp(-8.318 + 42.796 * a2_star - 59.938 * a2_star**2)
    else:
        p_value = 1 - math.exp(-13.436 + 101.14 * a2_star - 223.73 * a2_star**2)
    return a2, p_value


def Normality_Test(
    y: np.ndarray,
    x: np.ndarray,
    method: str = "shapiro-wilk",
    significance_level: float = 0.05,
) -> NormalityTestResult:
    """
    Test whether the sale ratios are normally distributed (Table 6.3).

    Many statistics assume a normal, bell-shaped distribution: the mean and
    the weighted mean, their confidence intervals, the standard deviation.
    The IAAO Standard on Ratio Studies (2026) says "Tests of normality should
    be done to determine if parametric or non-parametric tests should be
    used" (Section 6, Principles), and that "ratio study samples typically do
    not conform to the normal distribution" (Section 6.2). When the ratios
    are not normally distributed, rely on the distribution-free measures:
    the median, the COD and their confidence intervals, and rank-based tests
    such as ``Kruskal_Wallis``.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series).
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    method : {"shapiro-wilk", "dagostino-pearson", "anderson-darling"}, default "shapiro-wilk"
        One of Table 6.3's three tests. Any capitalization, with spaces,
        hyphens or underscores; ``"shapiro"``, ``"dagostino"`` and
        ``"anderson"`` also work.
    significance_level : float, default 0.05
        Normality is rejected when the p-value is below this (Table 10.2
        uses 0.05).

    Returns
    -------
    NormalityTestResult
        ``statistic``, ``p_value``, ``conclusion`` (``'Not normally
        distributed'`` or ``'No significant departure from normality'``),
        ``n_sales``, ``skewness`` and ``excess_kurtosis``; ``print(result)``
        shows a report. When the test can't be done, the statistic and the
        p-value are NaN and the conclusion says why.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``method`` is not one of the three tests; or if
        ``significance_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in the module documentation); if there are too few sales for
        the test (Shapiro-Wilk needs 3, the other two 8) or every ratio is
        the same; or, for Shapiro-Wilk, with more than 5,000 sales (see
        Notes).

    See Also
    --------
    NormalityTestResult : the result.
    Kruskal_Wallis : a rank-based test that needs no normal distribution.

    Notes
    -----
    Each test checks the hypothesis that the ratios come from a normal
    distribution; a small p-value is evidence that they don't.

    * **Shapiro-Wilk W** (the default): how closely the sorted ratios follow
      the pattern a normal distribution would give; W is near 1 for normal
      data. Calculated with Royston's approximation (1992, 1995; Algorithm
      AS R94, which scipy and R use), fitted up to 5,000 sales. In
      simulations it was the most powerful of the three and held its 5%
      significance level. The library calculates it itself, in double
      precision: scipy 1.10 computes it partly in single precision, which
      moved p-values by up to 0.004 at 5,000 sales.
    * **D'Agostino-Pearson K2**: a test of skewness (D'Agostino 1970) and a
      test of kurtosis (Anscombe and Glynn 1983) combined, as in D'Agostino,
      Belanger and D'Agostino (1990), which the Standard cites:
      K2 = z_skewness**2 + z_kurtosis**2, a chi-square with 2 degrees of
      freedom. Its kurtosis part is approximate below 20 sales. Calculated
      by scipy.
    * **Anderson-Darling A2**: how far the ratios' distribution is from the
      normal one, giving extra weight to the tails, with the mean and the
      standard deviation (n - 1) estimated from the ratios::

          A2 = -n - (1/n) * sum over i of (2i - 1) * [ln F(z_i) + ln(1 - F(z_(n+1-i)))]
          a  = A2 * (1 + 0.75/n + 2.25/n**2)            (the modified statistic, A2*)
          p  = exp(1.2937 - 5.709 a + 0.0186 a**2)       when a >= 0.6
               exp(0.9177 - 4.279 a - 1.38 a**2)         when 0.34 <= a < 0.6
               1 - exp(-8.318 + 42.796 a - 59.938 a**2)  when 0.2 <= a < 0.34
               1 - exp(-13.436 + 101.14 a - 223.73 a**2) when a < 0.2

      with z_1 <= ... <= z_n the standardized ratios and F the normal
      distribution; the p-value formulas are D'Agostino and Stephens'
      (1986), which the Standard cites. (The first one has its minimum at
      a = 153.47, p about 1e-190, and turns up after it, so larger values
      use the minimum.) The library calculates this test itself, so its
      result doesn't change with the version of scipy, whose own version
      changes between releases.

    ``skewness = m3 / m2**1.5`` and ``excess_kurtosis = m4 / m2**2 - 3``,
    with m_k the average of (ratio - mean ratio)**k: both 0 for a normal
    distribution. Sale ratios usually have a long tail of high ratios
    (positive skewness). Excel's SKEW and KURT add small-sample corrections,
    so they differ a little.

    With many sales, any of these tests finds almost any departure from
    normality significant: in simulations, 5,000 nearly normal ratios
    (skewness 0.15) were rejected by all three. Read the skewness and the
    kurtosis alongside the conclusion. Above 5,000 sales the Shapiro-Wilk
    p-value is an approximation, and the function warns.

    For the Standard's 54 example sales (Table 10.1) none of the three
    rejects normality at 0.05: W = 0.9702 (p = 0.1962), K2 = 4.2015
    (p = 0.1224), A2 = 0.4718 (p = 0.2352). The Standard's Table 10.2 says
    "Reject" for these sales without saying which test it used; none of
    these three gives that result.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.Normality_Test(sales["Sale_Price"], sales["Appraised_Value"])
    >>> result.conclusion, result.p_value
    >>> print(result)
    >>> rem.Normality_Test(sales["Sale_Price"], sales["Appraised_Value"],
    ...                    method="anderson-darling")
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    test = _check_normality_method(method)
    name, _, minimum_sales = _NORMALITY_TESTS[test]

    y, x = _sales_pairs(y, x)
    ratio = x / y
    n_sales = len(ratio)
    skewness, excess_kurtosis = _skewness_and_excess_kurtosis(ratio)

    def not_calculated(conclusion: str, reason: str) -> NormalityTestResult:
        _warn_data_quality(f"{reason} Its statistic and p-value are left empty (NaN).")
        return NormalityTestResult(
            name, math.nan, math.nan, significance_level, conclusion, n_sales, skewness, excess_kurtosis
        )

    if n_sales < minimum_sales:
        return not_calculated(
            f"Not enough sales (fewer than {minimum_sales})",
            f"The {name} test of normality needs at least {minimum_sales} sales, but only {n_sales} usable "
            f"sales were given.",
        )
    if np.all(ratio == ratio[0]):
        return not_calculated(
            _NORMALITY_IDENTICAL_RATIOS,
            f"Every sale ratio is the same ({ratio[0]:.4f}), so there is no distribution to test (identical "
            "ratios are unusual in real data; please check them, for example for sales chasing). The "
            f"{name} test is not calculated.",
        )

    from scipy import stats  # imported here: slow to import (see top of module)

    if test == "shapirowilk":
        statistic, p_value = _shapiro_wilk(ratio)
    elif test == "dagostinopearson":
        with warnings.catch_warnings():
            # scipy's note that the kurtosis part is approximate below 20 sales is in the docstring
            warnings.simplefilter("ignore")
            statistic, p_value = stats.normaltest(ratio)
    else:
        statistic, p_value = _anderson_darling(ratio)
    if test == "shapirowilk" and n_sales > _SHAPIRO_WILK_MAXIMUM_SALES:
        _warn_data_quality(
            f"With more than 5,000 sales ({n_sales:,}), the Shapiro-Wilk p-value is an approximation (Royston's "
            "method is fitted up to 5,000 sales), and with this many sales almost any departure from normality is "
            "significant. Read the skewness and excess kurtosis too, or use method='dagostino-pearson'."
        )

    conclusion = _NORMALITY_REJECTED if p_value < significance_level else _NORMALITY_NOT_REJECTED
    return NormalityTestResult(
        name, float(statistic), float(p_value), significance_level, conclusion, n_sales, skewness, excess_kurtosis
    )


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

    Examples
    --------
    With the 54 example sales of the Standard's Table 10.1:

    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> result = rem.VEI(sale_prices, appraised_values)
    >>> result
    VEIResult(vei=48.53, ci_lower=29.52, ci_upper=67.54, scenario=4, conclusion='Progressive')
    >>> print(result)                  # plain-English report, with the group table
    >>> result.groups[["Group", "Sales", "Median_Ratio", "COD"]]
    >>> result.plot()                  # the test and the value groups, charted
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
        The lower confidence limit, in percent (29.52 means 29.52%). NaN when
        there are fewer than 20 sales.

    Raises
    ------
    ValueError
        As ``VEI``: if the inputs differ in length, are empty, are not
        numbers, or have no usable sales; if ``confidence_level`` is not
        between 0 and 1; or if ``ranking_method`` is not "R6" or "R7".

    Warns
    -----
    DataQualityWarning
        As ``VEI``: if sales are left out (including sales with an appraised
        value of 0 or less), or if fewer than 20 usable sales remain.

    Notes
    -----
    The interval of Step 6 of the IAAO Standard on Ratio Studies (2026),
    Appendix E. For the Standard's 54 example sales (Table 10.1), the 90%
    interval is 29.52% to 67.54% (Tables E.6.2 and 10.2).

    Each call runs the whole ``VEI``. To get several parts, call ``VEI``
    once and read ``ci_lower``, ``ci_upper`` and ``conclusion`` from its
    result.

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
        The upper confidence limit, in percent (67.54 means 67.54%). NaN when
        there are fewer than 20 sales.

    Raises
    ------
    ValueError
        As ``VEI``: if the inputs differ in length, are empty, are not
        numbers, or have no usable sales; if ``confidence_level`` is not
        between 0 and 1; or if ``ranking_method`` is not "R6" or "R7".

    Warns
    -----
    DataQualityWarning
        As ``VEI``: if sales are left out (including sales with an appraised
        value of 0 or less), or if fewer than 20 usable sales remain.

    Notes
    -----
    The interval of Step 6 of the IAAO Standard on Ratio Studies (2026),
    Appendix E. For the Standard's 54 example sales (Table 10.1), the 90%
    interval is 29.52% to 67.54% (Tables E.6.2 and 10.2).

    Each call runs the whole ``VEI``. To get several parts, call ``VEI``
    once and read ``ci_lower``, ``ci_upper`` and ``conclusion`` from its
    result.

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

    Raises
    ------
    ValueError
        As ``VEI``: if the inputs differ in length, are empty, are not
        numbers, or have no usable sales; if ``confidence_level`` is not
        between 0 and 1; or if ``ranking_method`` is not "R6" or "R7".

    Warns
    -----
    DataQualityWarning
        As ``VEI``: if sales are left out (including sales with an appraised
        value of 0 or less), or if fewer than 20 usable sales remain.

    Notes
    -----
    Each call runs the whole ``VEI``. To get several parts, call ``VEI``
    once and read ``ci_lower``, ``ci_upper`` and ``conclusion`` from its
    result.

    Examples
    --------
    >>> from pyRealEstate import RealEstateMetrics as rem
    >>> rem.VEI_Conclusion(sales["Sale_Price"], sales["Appraised_Value"])
    'No significant progressivity or regressivity'
    """
    return VEI(Sale_Amount, Appr_Value, confidence_level, ranking_method).conclusion
