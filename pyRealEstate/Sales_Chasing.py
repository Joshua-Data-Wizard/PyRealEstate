"""
Sales-chasing detection: Appendix B of the IAAO Standard on Ratio Studies
(2026).

Sales chasing is revaluing the parcels that sold from their sale prices (or
listing prices), so that sold parcels are valued differently from unsold
ones. A ratio study of those sales then looks more uniform, and closer to
market value, than the assessment roll really is. The Standard asks
valuation officers and oversight agencies to "consider implementing at least
one of the detection procedures outlined in this Appendix".

The Appendix's techniques, and where they are in the library:

* B.1, comparison of value changes: ``Value_Change_Comparison`` (the
  year-to-year value changes of sold vs unsold parcels). Its result's
  ``plot()`` draws the box plots B.1 suggests.
* B.2, comparison of unit values: ``Unit_Value_Comparison`` (e.g. value per
  square foot of sold vs unsold parcels), with the same ``plot()``.
* B.4, comparison of observed versus expected distribution of ratios:
  ``Ratio_Concentration``. It needs only a ratio study's sale prices and
  appraised values, so oversight agencies can use it too.
* B.5, mass appraisal techniques: run the ratio-study functions of
  ``RealEstateMetrics`` on unsold parcels, with an independent model's values
  in place of the sale prices (``y``).
* B.6, independent value chasing: ``RealEstateMetrics.Mann_Whitney``, with
  the ratios of independently valued and of sold properties as its two groups.
* B.3 (split sample) is not in the library yet.

Argument conventions. For sale ratios (B.4), as in ``RealEstateMetrics``::

    y : sale prices
    x : appraised / model-predicted values
    sale ratio = appraised value / sale price

For sold vs unsold parcels (B.1, B.2), one entry per parcel of the stratum,
paired by position: the values, and ``sold``, True for each parcel that sold
(False otherwise).

Data checks, the same policy as ``RealEstateMetrics`` (see "Data checks"
there): what cannot be used (it cannot form a sale ratio, a value change or a
value per unit) is left out, unusual appraised or current values of 0 or less
are kept, each with a ``DataQualityWarning``, and impossible inputs raise a
``ValueError``.
"""

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib is imported inside the functions that draw (it is slow to import).
if TYPE_CHECKING:  # for editors and type checkers only; never imported at run time
    import matplotlib.axes

from ._data_checks import (
    DataQualityWarning,  # so users can import it from this module too
    _describe_rows,
    _problem_lines,
    _row_labels,
    _warn_data_quality,
)
from .RealEstateMetrics import (
    _MW_NOT_CALCULATED,
    _check_fraction,
    _format_p_value,
    _mann_whitney_test,
    _percent,
    _sales_pairs,
    _warn_small_mann_whitney_groups,
)
from ._plotting import (
    _MINUS,
    _PLOT_STYLE,
    _add_title,
    _compact_dollars,
    _fit_title,
    _new_axes,
    _number,
    _style_axes,
)

# ---------------------------------------------------------------------------
# B.4: comparison of observed versus expected distribution of ratios
# ---------------------------------------------------------------------------

_B4_MINIMUM_SALES = 30  # B.4: "the method should not be employed for sample sizes less than 30"
_B4_RANGE = 0.02  # B.4: ratios within +-2% of a measure of central tendency
_B4_SD_PER_COD = 1.25  # Table B.4.1: standard deviation = 1.25 x COD (for a normal distribution; exactly sqrt(pi/2))

# The conclusions returned by Ratio_Concentration
_B4_INVESTIGATE = "Concentration at or above the critical proportion: investigate for sales chasing"
_B4_NO_CONCENTRATION = "No unusual concentration of ratios"
_B4_NOT_ENOUGH_SALES = f"Not enough sales (fewer than {_B4_MINIMUM_SALES})"


@dataclass(frozen=True, repr=False)
class RatioConcentrationResult:
    """
    The result of ``Ratio_Concentration``: are the sale ratios bunched more
    tightly than honest appraisal allows? (IAAO Standard on Ratio Studies
    2026, Appendix B.4, Table B.4.1.)

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    n_sales : int
        The number of usable sales.
    median_ratio : float
        The median sale ratio, the first center checked.
    cod : float
        The sales' coefficient of dispersion, in percent (as ``COD``).
    share_near_median : float
        The share of the sale ratios within +-2% of the median ratio
        (median x 0.98 to median x 1.02), e.g. 0.20 for 20%.
    share_near_100 : float
        The share of the sale ratios within +-2% of 100% (0.98 to 1.02),
        where ratios gather when sales are appraised at their sale price.
    minimum_cod : float
        The minimum achievable COD assumed, in percent.
    expected_proportion : float
        The share expected within +-2% of the center if the ratios were
        normally distributed with that COD (Table B.4.1's last column).
    critical_proportion : float
        Table B.4.1's critical proportion: the whole percent just above the
        expected proportion, e.g. 0.32 for a minimum achievable COD of 4%.
    conclusion : str
        ``'Concentration at or above the critical proportion: investigate
        for sales chasing'`` (either share is at or above the critical
        proportion), ``'No unusual concentration of ratios'``, or ``'Not
        enough sales (fewer than 30)'``.

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Ratio_Concentration(sales["Sale_Price"], sales["Appraised_Value"])
    >>> result.share_near_median, result.share_near_100, result.critical_proportion
    >>> result.conclusion
    >>> print(result)                  # plain-English report
    """

    n_sales: int
    median_ratio: float
    cod: float
    share_near_median: float
    share_near_100: float
    minimum_cod: float
    expected_proportion: float
    critical_proportion: float
    conclusion: str

    def __repr__(self) -> str:
        return (
            f"RatioConcentrationResult(share_near_median={self.share_near_median:.4f}, "
            f"share_near_100={self.share_near_100:.4f}, critical_proportion={self.critical_proportion:g}, "
            f"conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        low, high = 1 - _B4_RANGE, 1 + _B4_RANGE
        lines = [
            "Concentration of sale ratios: a check for sales chasing",
            "(IAAO Standard on Ratio Studies, 2026: Appendix B.4)",
            f"{self.n_sales} sales; median ratio {self.median_ratio:.4f}; COD {self.cod:.2f}",
            f"Within +-2% of the median ratio ({low * self.median_ratio:.4f} to {high * self.median_ratio:.4f}): "
            f"{self._sales_text(self.share_near_median)}",
            f"Within +-2% of 100% ({low:.4f} to {high:.4f}): {self._sales_text(self.share_near_100)}",
            f"Critical proportion for a minimum achievable COD of {self.minimum_cod:g}%: "
            f"{100 * self.critical_proportion:g}% (Table B.4.1; {100 * self.expected_proportion:.1f}% expected for "
            "normally distributed ratios)",
            f"Conclusion: {self.conclusion}",
        ]
        if self.conclusion == _B4_INVESTIGATE:
            lines.append(
                "The Standard asks for further investigation before concluding that sales chasing has occurred."
            )
        return "\n".join(lines)

    def _sales_text(self, share: float) -> str:
        """'8 sales (20.0%)'."""
        count = round(share * self.n_sales)
        return f"{count} sale{'' if count == 1 else 's'} ({100 * share:.1f}%)"


def _check_minimum_cod(minimum_cod: Any) -> None:
    """Raise a ValueError unless ``minimum_cod`` is a COD in percent (1 or more)."""
    if (
        isinstance(minimum_cod, bool)
        or not isinstance(minimum_cod, numbers.Real)
        or not math.isfinite(minimum_cod)
        or minimum_cod <= 0
    ):
        raise ValueError(
            "minimum_cod must be a positive number, in percent (for example 4.0 for a COD of 4%), "
            f"not {minimum_cod!r}."
        )
    if minimum_cod < 1:
        raise ValueError(
            "minimum_cod is in percent (4.0 means a COD of 4%), and a COD below 1% is not achievable for real "
            f"property: {minimum_cod!r} looks like a fraction. For a minimum achievable COD of 4%, use 4.0."
        )


def _critical_concentration(minimum_cod: float) -> Tuple[float, float]:
    """
    Table B.4.1 for one minimum achievable COD: the share of ratios expected
    within +-2% of the center if they were normally distributed with that
    COD, and the critical proportion, "the integer immediately exceeding the
    expected proportion" (in percent), as a fraction.
    """
    standard_deviation = _B4_SD_PER_COD * minimum_cod / 100  # of the ratios, e.g. 0.05 for a COD of 4%
    z = _B4_RANGE / standard_deviation  # the +-2% range in standard deviations
    expected = math.erf(z / math.sqrt(2))  # the chance a standard normal value is between -z and z
    critical = (math.floor(round(100 * expected, 10)) + 1) / 100
    return expected, critical


def _share_within_range(ratio: np.ndarray, center: float) -> float:
    """
    The share of the ratios within +-2% of ``center`` (center x 0.98 to
    center x 1.02, edges included). Distances are compared to 10 decimals, so
    a ratio exactly on an edge counts: in floating point, |0.98 - 1| is
    0.020000000000000018.
    """
    distance = np.round(np.abs(ratio - center), 10)
    return float(np.mean(distance <= round(_B4_RANGE * abs(center), 10)))


def Ratio_Concentration(y: np.ndarray, x: np.ndarray, minimum_cod: float = 4.0) -> RatioConcentrationResult:
    """
    Are the sale ratios bunched more tightly than honest appraisal allows?

    A check for sales chasing from the IAAO Standard on Ratio Studies (2026),
    Appendix B.4, "Comparison of observed versus expected distribution of
    ratios". When the parcels that sold are revalued from their sale prices,
    their ratios pile up close to one value. This function counts the share
    of sale ratios within +-2% of two centers named by the Standard, the
    median ratio and 100%, and compares each share with the critical
    proportion of Table B.4.1: the most that could be expected even from a
    very uniform assessment, one whose COD is the minimum achievable. A share
    at or above it calls for investigation.

    It needs only the sale prices and appraised values of a ratio study, so
    oversight agencies without access to every parcel's values can use it
    too (Appendix B).

    Parameters
    ----------
    y : array-like
        Sale prices (numpy array, list, or pandas Series), ideally adjusted
        for time (B.4 assumes "sales that have been properly adjusted for time
        and other factors").
    x : array-like
        Appraised / model-predicted values, in the same order as ``y``.
    minimum_cod : float, default 4.0
        The lowest COD (in percent) achievable for this property type, class
        or stratum. The default, 4%, is the Standard's example: a COD "many
        valuers consider ... to be unachievable" except in highly constrained
        markets, giving a "very conservative" critical proportion of 32%.
        When the minimum achievable COD is known to be higher, a higher value
        gives a lower critical proportion and catches more (Table B.4.1: 8%
        gives 16%).

    Returns
    -------
    RatioConcentrationResult
        The shares within +-2% of the median ratio (``share_near_median``) and
        of 100% (``share_near_100``), the ``expected_proportion`` and
        ``critical_proportion`` for ``minimum_cod``, the ``conclusion``, and
        the sales' ``n_sales``, ``median_ratio`` and ``cod``.
        ``print(result)`` shows a plain-English report.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, are not numbers, or have
        no usable sales; or if ``minimum_cod`` is not a number of 1 or more
        (it is in percent: 4.0 means 4%).

    Warns
    -----
    DataQualityWarning
        If sales are left out or appraised values look unusual (see "Data
        checks" in ``RealEstateMetrics``), or if there are fewer than 30
        usable sales.

    See Also
    --------
    RealEstateMetrics.COD : the sales' uniformity.
    RealEstateMetrics.Mann_Whitney : B.6, sold vs independently valued
        properties.

    Notes
    -----
    The calculation (Appendix B.4 and Table B.4.1)::

        ratio                = appraised value / sale price
        share near median    = share of ratios between median x 0.98 and median x 1.02
        share near 100%      = share of ratios between 0.98 and 1.02
        standard deviation   = 1.25 x minimum_cod          (a normal distribution, mean ratio 100%)
        z                    = 2% / standard deviation
        expected proportion  = chance that a standard normal value is between -z and z
        critical proportion  = the whole percent immediately above the expected proportion
        conclusion           = "investigate" when either share >= the critical proportion

    For the default minimum COD of 4%: standard deviation 5%, z = 0.4,
    expected 31.08%, critical 32%. The function reproduces all 10 critical
    proportions of Table B.4.1 (two of the table's other columns have
    misprints in the rows for 7% and 12%, which do not affect them). The
    factor 1.25 is the Standard's; for a normal distribution it is exactly
    sqrt(pi/2) = 1.2533, which would turn the 6% row's 22% into 21%.

    **Why two centers.** The Standard counts ratios "within +-2% of a measure
    of central tendency, such as the median or a mode, or 100%". Sales
    appraised at their sale price have ratios of exactly 1.00, which is not
    the median when the overall level is not 100%. In a simulation (level
    0.92, 30% of 1,000 sales chased), the share near the median caught it in
    54% of samples and the share near 100% in all of them, without false
    alarms on honest data. A search for the densest window anywhere
    (a "mode") is not done: Table B.4.1 assumes one fixed center, and in
    small samples such a search raises false alarms.

    The +-2% is relative to the center (median x 0.98 to x 1.02), like the
    COD it is compared with, so the check works the same at any level of
    appraisal; edges are included.

    **A screening check, not proof.** "Even when critical proportions of
    ratios shown in Table B.4.1 are exceeded, further investigation should
    be conducted before concluding that sales chasing has occurred." It is
    only as good as ``minimum_cod``: when a stratum's honest COD is close to
    it, honest data can cross the line (measured: a COD near 4.5% crossed
    32% in 41% of samples of 30 sales, 21% of 100), and with the default 4%
    modest chasing is missed (20% of the sales chased was caught in about
    35-47% of samples). A low COD with a high concentration around one value,
    or a bimodal distribution of ratios, deserves a closer look.

    **Sample size.** The Standard: "the method should not be employed for
    sample sizes less than 30". With fewer than 30 usable sales, the shares
    are still reported, the conclusion is "Not enough sales (fewer than
    30)", and a warning says why. (Table B.4.1's note assumes a sample
    "greater than 30"; this function follows the text, so 30 sales are
    enough.)

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Ratio_Concentration(sales["Sale_Price"], sales["Appraised_Value"])
    >>> print(result)                         # plain-English report
    >>> result.conclusion
    >>> sc.Ratio_Concentration(sales["Sale_Price"], sales["Appraised_Value"],
    ...                        minimum_cod=8.0)  # a stratum where 8% is the lowest COD achievable
    """
    _check_minimum_cod(minimum_cod)
    sale_price, appraised_value = _sales_pairs(y, x)
    ratio = appraised_value / sale_price
    n_sales = len(ratio)

    # The median ratio and the COD, calculated as RealEstateMetrics.Median_Ratio and COD do
    median_ratio = np.median(ratio)
    cod = 100 * ((np.sum(np.abs(ratio - median_ratio)) / n_sales) / median_ratio)

    share_near_median = _share_within_range(ratio, median_ratio)
    share_near_100 = _share_within_range(ratio, 1.0)
    expected, critical = _critical_concentration(minimum_cod)

    if n_sales < _B4_MINIMUM_SALES:
        _warn_data_quality(
            f"Appendix B.4's check needs at least {_B4_MINIMUM_SALES} sales (the Standard: it \"should not be "
            f"employed for sample sizes less than {_B4_MINIMUM_SALES}\"), but there are {n_sales} usable sales. "
            "The concentrations are reported without a conclusion."
        )
        conclusion = _B4_NOT_ENOUGH_SALES
    elif max(share_near_median, share_near_100) >= critical:
        conclusion = _B4_INVESTIGATE
    else:
        conclusion = _B4_NO_CONCENTRATION

    return RatioConcentrationResult(
        n_sales=n_sales,
        median_ratio=float(median_ratio),
        cod=float(cod),
        share_near_median=share_near_median,
        share_near_100=share_near_100,
        minimum_cod=float(minimum_cod),
        expected_proportion=expected,
        critical_proportion=critical,
        conclusion=conclusion,
    )


# ---------------------------------------------------------------------------
# B.1 and B.2: sold vs unsold parcels (value changes, values per unit)
# ---------------------------------------------------------------------------

_B1_TOLERANCE = 5.0  # B.1: a difference in value changes "greater than 5%" (percentage points) ...
_B1_SMALL_CHANGE = 5.0  # ... "if value changes for sold properties are less than 5%" ...
_B1_SMALL_CHANGE_TOLERANCE = 3.0  # ... "reduced to 3%"; at exactly 5% too, as B.1's own example (user decision)
_B2_TOLERANCE = 5.0  # B.1's tolerance "applies to other detection techniques": 5% of the unsold value per unit
_UNCHANGED_NOTE = 0.5  # the report notes a group of which half or more did not change in value

# The conclusions returned by Value_Change_Comparison and Unit_Value_Comparison
_SU_LIKELY = "Significant difference above the tolerance: sales chasing likely"
_SU_INVESTIGATE = "Significant difference above the tolerance: investigate for sales chasing"
_SU_WITHIN = "Significant difference within the tolerance"
_SU_NOT_SIGNIFICANT = "No significant difference"
_SU_NOT_ENOUGH = "Not enough parcels (both sold and unsold parcels are needed)"
_SU_NO_TOLERANCE = "Significant difference (the tolerance cannot be applied)"

_CENTERS = ("median", "mean")


class _Argument(NamedTuple):
    """How messages name one of a comparison's value arguments."""

    argument: str  # e.g. "prior_value"
    name: str  # e.g. "prior value"
    plural: str  # e.g. "prior values"


class _Comparison(NamedTuple):
    """The wording and rules of B.1 or B.2."""

    technique: str  # "B.1" or "B.2"
    title: str  # "Value changes"
    measure: str  # what is compared, for one parcel: "value change"
    first: _Argument  # the function's first argument ...
    second: _Argument  # ... and its second
    first_is_denominator: bool  # B.1 divides by the prior value (first), B.2 by the units (second)
    columns: Tuple[str, str]  # the groups table's median and mean columns
    parcel_column: str  # the parcels table's column for each parcel's value change (or value per unit)
    axis_label: str  # the chart's value axis
    above_tolerance: str  # the conclusion for a significant difference above the tolerance


_B1 = _Comparison(
    technique="B.1",
    title="Value changes",
    measure="value change",
    first=_Argument("prior_value", "prior value", "prior values"),
    second=_Argument("current_value", "current value", "current values"),
    first_is_denominator=True,
    columns=("Median_Change", "Mean_Change"),
    parcel_column="Value_Change",
    axis_label="Value change from the prior to the current value",
    above_tolerance=_SU_LIKELY,
)
_B2 = _Comparison(
    technique="B.2",
    title="Values per unit",
    measure="value per unit",
    first=_Argument("value", "value", "values"),
    second=_Argument("units", "unit count", "unit counts"),
    first_is_denominator=False,
    columns=("Median_Value_Per_Unit", "Mean_Value_Per_Unit"),
    parcel_column="Value_Per_Unit",
    axis_label="Value per unit",
    above_tolerance=_SU_INVESTIGATE,
)
_COMPARISONS = {comparison.technique: comparison for comparison in (_B1, _B2)}


@dataclass(frozen=True, eq=False, repr=False)
class SoldUnsoldResult:
    """
    The result of ``Value_Change_Comparison`` (IAAO Standard on Ratio Studies
    2026, Appendix B.1) or ``Unit_Value_Comparison`` (Appendix B.2): are sold
    and unsold parcels valued the same way?

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    technique : str
        ``'B.1'`` (value changes) or ``'B.2'`` (values per unit).
    center : str
        ``'median'`` or ``'mean'``: which center ``difference`` compares.
    groups : pandas.DataFrame
        Two rows, ``Sold`` and ``Unsold``: ``Group``, ``Parcels``, the
        median and mean (``Median_Change`` / ``Mean_Change`` in percent for
        B.1; ``Median_Value_Per_Unit`` / ``Mean_Value_Per_Unit`` for B.2),
        and for B.1 ``Unchanged``, the share of parcels whose value did not
        change.
    difference : float
        B.1: the sold parcels' center minus the unsold parcels', in
        percentage points. B.2: how much the sold parcels' center is above
        (+) or below (-) the unsold parcels', in percent. NaN when it cannot
        be calculated.
    tolerance : float
        The practical tolerance the difference is compared with, in either
        direction: B.1, 3 points when the sold parcels' change is 5% or less,
        else 5; B.2, 5%. NaN when it cannot be applied.
    statistic : float
        The Mann-Whitney U of the sold parcels: the number of pairs (one sold,
        one unsold parcel) in which the sold parcel's value change (or value
        per unit) is higher, a tie counting half.
    share_higher : float
        U / (sold x unsold): the share of those pairs.
    p_value : float
        The Mann-Whitney test's two-sided p-value (exact with 50 parcels or
        fewer). NaN when it cannot be calculated.
    p_value_method : str
        ``'exact'``, ``'normal approximation'`` or ``'not calculated'``.
    significance_level : float
        The significance level used for the conclusion, e.g. 0.05.
    conclusion : str
        ``'Significant difference above the tolerance: sales chasing
        likely'`` (B.1) or ``'... investigate for sales chasing'`` (B.2),
        ``'Significant difference within the tolerance'``, ``'No significant
        difference'``, ``'Not enough parcels (both sold and unsold parcels are
        needed)'``, or ``'Significant difference (the tolerance cannot be
        applied)'``.
    parcels : pandas.DataFrame
        One row per usable parcel, labeled like the input's rows (the pandas
        index, or the position): ``Sold`` (True / False) and the parcel's
        ``Value_Change`` in percent (B.1) or ``Value_Per_Unit`` (B.2). Sort it
        to find the parcels behind the outliers of ``plot()``.

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Value_Change_Comparison(
    ...     parcels["Prior_Value"], parcels["Current_Value"], parcels["Sold"]
    ... )
    >>> result.difference, result.tolerance, result.conclusion
    >>> result.groups                  # each group's parcels, median and mean change, share unchanged
    >>> print(result)                  # plain-English report
    >>> result.plot()                  # box plot of the sold and unsold parcels' changes
    >>> result.parcels.sort_values("Value_Change").tail()   # the largest changes
    """

    technique: str
    center: str
    groups: pd.DataFrame
    difference: float
    tolerance: float
    statistic: float
    share_higher: float
    p_value: float
    p_value_method: str
    significance_level: float
    conclusion: str
    parcels: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"SoldUnsoldResult(technique={self.technique!r}, difference={self.difference:.4f}, "
            f"tolerance={self.tolerance:g}, p_value={self.p_value:.4f}, conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        comparison = _COMPARISONS[self.technique]
        n_sold, n_unsold = (int(n) for n in self.groups["Parcels"])
        lines = [
            f"{comparison.title} of sold vs unsold parcels: a check for sales chasing",
            f"(IAAO Standard on Ratio Studies, 2026: Appendix {self.technique})",
            f"{n_sold + n_unsold} parcels: {n_sold} sold, {n_unsold} unsold",
        ]
        if self.conclusion == _SU_NOT_ENOUGH:
            lines.append(f"Conclusion: {self.conclusion}")
            return "\n".join(lines)

        formats = {column: "{:.2f}".format for column in comparison.columns}
        formats["Unchanged"] = "{:.1%}".format
        lines += [
            "",
            self.groups.to_string(index=False, formatters=formats),
            "",
            self._difference_text(),
            self._test_text(comparison),
            f"Conclusion at the {_percent(self.significance_level)} significance level: {self.conclusion}",
        ]
        if self.conclusion == _SU_LIKELY:
            lines.append(
                "Before acting, make sure parcels with physical changes, changes in use, or values changed on "
                "appeal were left out (B.1)."
            )
        if self.technique == "B.2" and self.conclusion not in (_SU_NOT_SIGNIFICANT, _SU_NOT_ENOUGH):
            lines.append(
                "Values per unit also differ when the parcels that sold differ from the rest (size, age, "
                "location): compare like with like."
            )
        lines += self._unchanged_notes()
        return "\n".join(lines)

    def _difference_text(self) -> str:
        """'Difference in median change: 7.50 points (sold higher); tolerance 5 points (...)'."""
        if math.isnan(self.difference):
            return (
                f"Difference in {self.center} value per unit: not calculated (the unsold parcels' "
                f"{self.center} is 0 or less)"
            )
        direction = "sold higher" if self.difference > 0 else "sold lower" if self.difference < 0 else "no difference"
        if self.technique == "B.1":
            which = "5% or less" if self.tolerance == _B1_SMALL_CHANGE_TOLERANCE else "above 5%"
            return (
                f"Difference in {self.center} change: {self.difference:.2f} points ({direction}); tolerance "
                f"{self.tolerance:g} points (the sold parcels' {self.center} change is {which})"
            )
        return (
            f"Difference in {self.center} value per unit: {self.difference:+.2f}% ({direction}); tolerance "
            f"{self.tolerance:g}%"
        )

    def _test_text(self, comparison: _Comparison) -> str:
        """The Mann-Whitney line of the report."""
        if math.isnan(self.p_value):
            return f"Mann-Whitney test: the p-value cannot be calculated: every {comparison.measure} is the same."
        return (
            f"Mann-Whitney test: U = {self.statistic:.1f}, p-value {_format_p_value(self.p_value)} "
            f"({self.p_value_method}); the sold parcel's {comparison.measure} is the higher one in "
            f"{self.share_higher:.1%} of the pairs (one sold, one unsold parcel)"
        )

    def _unchanged_notes(self) -> list:
        """B.1: a note for each group of which half or more did not change in value."""
        if "Unchanged" not in self.groups:
            return []
        notes = [
            f"Note: {share:.1%} of the {group.lower()} parcels did not change in value."
            for group, share in zip(self.groups["Group"], self.groups["Unchanged"])
            if share >= _UNCHANGED_NOTE
        ]
        if notes:
            notes.append(
                "B.1: when many unsold parcels did not change, \"the median value change in that group may not "
                "be meaningful, in which case the mean should be used\" (center=\"mean\"); a high share unchanged "
                "among unsold parcels but not sold ones is itself a sign of selective revaluation."
            )
        return notes

    def plot(
        self,
        title: Optional[str] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        show: bool = True,
    ) -> None:
        """
        Chart the comparison: a box plot of the sold and unsold parcels, as
        Appendix B.1 of the IAAO Standard on Ratio Studies (2026) suggests
        ("Visualizations, such as boxplots, can compare value change
        percentages across sold and unsold property groups").

        The sold parcels are the blue box (the group under test), the unsold
        parcels the gray box below it (the reference). The tolerance is shaded
        around the unsold parcels' center, so whether the sold center falls
        outside it shows at a glance. Both centers are labeled (with
        ``center="mean"``, the means are drawn as diamonds); every parcel is
        shown, the outliers as small circles. For B.1, 0% (no change) is
        marked, and each group's share unchanged is beside its box. The title
        states the verdict; the subtitle gives the difference, the tolerance
        and the Mann-Whitney p-value.

        Parameters
        ----------
        title : str, optional
            Your own title instead of the default ("Value changes, sold vs
            unsold parcels: <verdict>" or "Values per unit, ..."). The subtitle
            then starts with the verdict, so the chart still states it.
        ax : matplotlib.axes.Axes, optional
            Draw on these axes (e.g. one panel per neighborhood: B.1's
            "visualizations can be repeated across multiple data strata")
            instead of a new figure, leaving your figure's colors alone.
        show : bool, default True
            Display the chart. Use False to keep customizing it or to save
            it, for example with ``plt.savefig("sold_vs_unsold.png", dpi=300)``.

        Returns
        -------
        None
            The chart is drawn on a new figure, or on ``ax``.
            ``plt.gcf()`` returns the figure.

        Notes
        -----
        A sold center inside the shaded band is within the tolerance. Outside
        it, the conclusion also needs a significant Mann-Whitney test (the
        subtitle gives its p-value). When the tolerance cannot be applied (B.2
        with an unsold center of 0 or less) there is no band. With no sold or
        no unsold parcels there is nothing to compare, so the chart shows only
        its title and the reason.

        Dollar signs in the chart's text are escaped, because matplotlib reads
        text between two ``$`` signs as math.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from pyRealEstate import Sales_Chasing as sc
        >>> result = sc.Value_Change_Comparison(
        ...     parcels["Prior_Value"], parcels["Current_Value"], parcels["Sold"]
        ... )
        >>> result.plot()
        >>> result.plot(title="Neighborhood 12, 2026 roll", show=False)
        >>> plt.savefig("sold_vs_unsold.png", dpi=300)
        >>> names = ["North", "South"]                          # one panel per stratum
        >>> fig, axes = plt.subplots(len(names), 1, figsize=(9, 8), layout="constrained")
        >>> for name, ax in zip(names, axes):
        ...     stratum = parcels[parcels["Neighborhood"] == name]
        ...     sc.Value_Change_Comparison(stratum["Prior_Value"], stratum["Current_Value"],
        ...                                stratum["Sold"]).plot(title=name, ax=ax, show=False)
        """
        import matplotlib.pyplot as plt  # imported here: slow to import (see RealEstateMetrics)

        if self.conclusion == _SU_NOT_ENOUGH:
            _sold_unsold_plot_not_enough(self, title, ax)
        else:
            _sold_unsold_plot(self, title, ax)
        if show:
            plt.show()


# ---------------------------------------------------------------------------
# The sold vs unsold chart (SoldUnsoldResult.plot). The shared colors and
# layout are in _plotting.py.
# ---------------------------------------------------------------------------

# The verdict in the chart's title
_SU_VERDICTS = {
    _SU_LIKELY: "sales chasing likely",
    _SU_INVESTIGATE: "investigate for sales chasing",
    _SU_WITHIN: "difference within the tolerance",
    _SU_NOT_SIGNIFICANT: "no significant difference",
    _SU_NO_TOLERANCE: "significant difference",
    _SU_NOT_ENOUGH: "not enough parcels",
}


def _horizontal_boxes() -> dict:
    """
    The ``boxplot`` option for horizontal boxes: ``orientation="horizontal"``
    in newer matplotlib (where ``vert=False`` is deprecated from 3.11), or
    ``vert=False`` in older versions, which have no ``orientation``.
    """
    import inspect

    import matplotlib.axes  # imported here: slow to import (see RealEstateMetrics)

    if "orientation" in inspect.signature(matplotlib.axes.Axes.boxplot).parameters:
        return {"orientation": "horizontal"}
    return {"vert": False}


def _sold_unsold_value_text(result: SoldUnsoldResult, value: float) -> str:
    """
    One value change ("+11.48%") or value per unit ("\\$119.59") for the
    chart. The dollar sign is escaped: matplotlib reads text between two $
    signs as math.
    """
    if result.technique == "B.1":
        return _number(value, "+.2f", "%")
    return f"{_MINUS if value < 0 else ''}\\${abs(value):,.2f}"


def _sold_unsold_centers(result: SoldUnsoldResult) -> Tuple[float, float]:
    """The sold and unsold parcels' compared centers (medians, or means with center="mean")."""
    comparison = _COMPARISONS[result.technique]
    column = comparison.columns[0 if result.center == "median" else 1]
    sold_center, unsold_center = result.groups[column]
    return float(sold_center), float(unsold_center)


def _sold_unsold_facts(result: SoldUnsoldResult) -> List[str]:
    """The subtitle's parts: the difference and tolerance, the test, and any group mostly unchanged."""
    comparison = _COMPARISONS[result.technique]
    sold_center, unsold_center = _sold_unsold_centers(result)
    which = f"{result.center.capitalize()}{' change' if result.technique == 'B.1' else ''}"
    centers = (f"{which} {_sold_unsold_value_text(result, sold_center)} sold vs "
               f"{_sold_unsold_value_text(result, unsold_center)} unsold")
    if math.isnan(result.difference):
        difference = f"{centers}: relative difference not calculated (the unsold {result.center} is 0 or less)"
    elif result.technique == "B.1":
        difference = f"{centers}: {_number(result.difference, '+.2f')} points (tolerance {result.tolerance:g})"
    else:
        difference = f"{centers}: {_number(result.difference, '+.1f', '%')} (tolerance {result.tolerance:g}%)"

    if math.isnan(result.p_value):
        test = f"Mann-Whitney test not calculated (every {comparison.measure} is the same)"
    else:
        test = f"Mann-Whitney p {_format_p_value(result.p_value)}"
    facts = [difference, test]
    if "Unchanged" in result.groups:
        facts += [f"{share:.1%} of {group.lower()} parcels unchanged"
                  for group, share in zip(result.groups["Group"], result.groups["Unchanged"])
                  if share >= _UNCHANGED_NOTE]
    return facts


def _sold_unsold_draw(result: SoldUnsoldResult, ax: "matplotlib.axes.Axes") -> bool:
    """
    Draw the horizontal box plot: sold (blue) above unsold (gray), the
    tolerance shaded around the unsold center, the centers labeled. Returns
    whether the band was drawn (its label sits above the plot, below the
    title).
    """
    # Imported here: slow to import (see RealEstateMetrics).
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    style = _PLOT_STYLE
    comparison = _COMPARISONS[result.technique]
    b1 = result.technique == "B.1"
    _style_axes(ax, grid_axis="x")

    # Every parcel, sold above unsold
    measure = result.parcels[comparison.parcel_column].to_numpy()
    sold = result.parcels["Sold"].to_numpy()
    positions = [2, 1]
    boxes = ax.boxplot(
        [measure[sold], measure[~sold]], positions=positions, widths=0.5, patch_artist=True,
        medianprops={"linewidth": 2.2},
        whiskerprops={"color": style["context"], "linewidth": 1},
        capprops={"color": style["context"], "linewidth": 1},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                    "markeredgecolor": style["context"], "markeredgewidth": 0.7},
        **_horizontal_boxes(),
    )
    colors = [(style["blue"], style["blue"], style["blue_dark"]),
              (style["context"], style["context_dark"], style["context_dark"])]
    for box, median, (fill, edge, median_color) in zip(boxes["boxes"], boxes["medians"], colors):
        box.set_facecolor(mcolors.to_rgba(fill, 0.18 if fill == style["blue"] else 0.22))
        box.set_edgecolor(edge)
        box.set_linewidth(1)
        median.set_color(median_color)

    # The tolerance, shaded around the unsold center
    sold_center, unsold_center = _sold_unsold_centers(result)
    banded = not math.isnan(result.tolerance)
    if banded:
        if b1:
            low, high = unsold_center - result.tolerance, unsold_center + result.tolerance
            unit = " points"
        else:
            low, high = unsold_center * (1 - result.tolerance / 100), unsold_center * (1 + result.tolerance / 100)
            unit = ""
        percent = "" if b1 else "%"
        label = (f"tolerance: unsold {result.center} {_number(-result.tolerance, '+.0f', percent)} to "
                 f"{_number(result.tolerance, '+.0f', percent)}{unit}")
        ax.axvspan(low, high, color=style["band"], linewidth=0, zorder=0)
        ax.text((low + high) / 2, 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color=style["muted"])
    if b1:
        ax.axvline(0, color=style["axis"], linewidth=1, zorder=1)  # no change

    # The compared centers, labeled above their boxes (the means as diamonds)
    for position, center, text_color, marker_color in (
        (positions[0], sold_center, style["text"], style["blue_dark"]),
        (positions[1], unsold_center, style["text_secondary"], style["context_dark"]),
    ):
        if result.center == "mean":
            ax.scatter([center], [position], marker="D", s=46, color=marker_color, edgecolor=style["surface"],
                       linewidth=1.5, zorder=4)
        ax.annotate(f"{result.center} {_sold_unsold_value_text(result, center)}", (center, position + 0.25),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=text_color)

    # Each group's parcels (and, for B.1, share unchanged) beside its box
    labels = []
    for row in result.groups.itertuples(index=False):
        label = f"{row.Group}\n{row.Parcels:,} parcel{'' if row.Parcels == 1 else 's'}"
        if b1:
            label += f"\n{row.Unchanged:.1%} unchanged"
        labels.append(label)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0.4, 2.75)
    if b1:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: "0%" if value == 0 else _number(value, "+.0f", "%")))
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: (_MINUS if value < 0 else "") + _compact_dollars(abs(value))))
    ax.set_xlabel(comparison.axis_label, color=style["text_secondary"], fontsize=10)
    return banded


def _sold_unsold_plot(
    result: SoldUnsoldResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]
) -> None:
    """``SoldUnsoldResult.plot()`` when both groups have parcels."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 3.9))
    banded = _sold_unsold_draw(result, ax)
    comparison = _COMPARISONS[result.technique]
    verdict = _SU_VERDICTS[result.conclusion]
    facts = _sold_unsold_facts(result)
    if title is None:
        title = f"{comparison.title}, sold vs unsold parcels: {verdict}"
    else:
        facts = [verdict[0].upper() + verdict[1:]] + facts
    title, subtitle = _fit_title(ax, own_figure, title, facts)
    _add_title(ax, title, subtitle, offset=4 + (14 if banded else 0))
    if own_figure:  # on the caller's axes it could land on another panel
        # The source, right-aligned with the plot, just below the axis label
        # (whatever its font, so the two never overlap).
        ax.annotate(f"IAAO Standard on Ratio Studies (2026), Appendix {result.technique}", xy=(1, 0),
                    xycoords=("axes fraction", ax.xaxis.label), xytext=(0, -3), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=_PLOT_STYLE["muted"])


def _sold_unsold_plot_not_enough(
    result: SoldUnsoldResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]
) -> None:
    """With no sold or no unsold parcels there is nothing to compare: only the title and the reason."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 1.0))
    ax.axis("off")
    n_sold, n_unsold = (int(n) for n in result.groups["Parcels"])
    reason = f"{n_sold:,} sold and {n_unsold:,} unsold parcels; the comparison needs both"
    if title is None:
        title, facts = f"{_COMPARISONS[result.technique].title}, sold vs unsold parcels: not enough parcels", [reason]
    else:
        facts = [result.conclusion, reason]
    title, subtitle = _fit_title(ax, own_figure, title, facts)
    _add_title(ax, title, subtitle)


def _check_center(center: Any) -> str:
    """'median' or 'mean' (any capitalization and spacing); anything else raises a ValueError."""
    if isinstance(center, str) and center.strip().lower() in _CENTERS:
        return center.strip().lower()
    raise ValueError(f"center must be 'median' or 'mean', not {center!r}.")


def _sold_flags(flags: np.ndarray, row_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Whether each parcel sold, and which flags are missing (None, NaN, pd.NA).
    True / False and 1 / 0 are accepted; anything else raises a ValueError.
    """
    if flags.dtype == bool:
        return flags.copy(), np.zeros(len(flags), dtype=bool)
    missing = np.asarray(pd.isna(flags), dtype=bool)
    present = flags[~missing]
    if pd.api.types.is_numeric_dtype(flags):
        valid = np.isin(present, (0, 1))
    else:  # objects: Python or numpy booleans, or numbers
        valid = np.array(
            [isinstance(f, (bool, np.bool_)) or (isinstance(f, numbers.Real) and f in (0, 1)) for f in present],
            dtype=bool,
        )
    if not valid.all():
        position = np.flatnonzero(~missing)[np.argmin(valid)]
        flag = flags[position]
        flag = flag.item() if isinstance(flag, np.generic) else flag
        raise ValueError(
            f"sold must say whether each parcel sold, as True / False (or 1 / 0); found {flag!r} "
            f"({_describe_rows(row_labels[[position]])}). For codes such as 'Y' / 'N', compare first, "
            "e.g. parcels[\"Sold\"] == \"Y\"."
        )
    is_sold = np.zeros(len(flags), dtype=bool)
    is_sold[~missing] = present.astype(bool)
    return is_sold, missing


def _parcel_values(
    first: Any, second: Any, sold: Any, comparison: _Comparison
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Check the parcels, following the ratio functions' data policy, and return
    (numerator, denominator, sold, row labels) for the usable ones: B.1
    (current value, prior value), B.2 (value, units). A parcel whose denominator is missing,
    infinite or 0 or less, whose numerator is missing or infinite, or whose
    sold flag is missing is left out with a warning; a numerator of 0 or less
    is kept with a warning; impossible inputs raise a ValueError.
    """
    first_values, second_values = pd.Series(first).to_numpy(), pd.Series(second).to_numpy()
    flags = pd.Series(sold).to_numpy()
    if len(first_values) != len(second_values):
        raise ValueError(
            f"There are {len(first_values)} {comparison.first.plural} ({comparison.first.argument}) but "
            f"{len(second_values)} {comparison.second.plural} ({comparison.second.argument}); each parcel needs "
            "exactly one of each."
        )
    n_parcels = len(first_values)
    if len(flags) != n_parcels:
        raise ValueError(
            f"There are {n_parcels} parcels but {len(flags)} sold flags (sold); each parcel needs exactly one."
        )
    if n_parcels == 0:
        raise ValueError(
            f"No parcels were provided: {comparison.first.argument}, {comparison.second.argument} and sold "
            "are empty."
        )
    for values, argument in ((first_values, comparison.first), (second_values, comparison.second)):
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{argument.plural.capitalize()} ({argument.argument}) must be numbers.")

    row_labels = _row_labels((first, second, sold), n_parcels)
    is_sold, missing_flag = _sold_flags(flags, row_labels)
    if comparison.first_is_denominator:
        (denominator, below), (numerator, above) = (first_values, comparison.first), (second_values, comparison.second)
    else:
        (denominator, below), (numerator, above) = (second_values, comparison.second), (first_values, comparison.first)
    denominator, numerator = denominator.astype(float), numerator.astype(float)

    # Parcels that cannot give a value change (or a value per unit) are left out.
    cannot_use = [
        (f"missing {below.name}", np.isnan(denominator)),
        (f"infinite {below.name}", np.isinf(denominator)),
        (f"{below.name} of 0", denominator == 0),
        (f"negative {below.name}", np.isfinite(denominator) & (denominator < 0)),
        (f"missing {above.name}", np.isnan(numerator)),
        (f"infinite {above.name}", np.isinf(numerator)),
        ("missing sold flag", missing_flag),
    ]
    left_out = np.logical_or.reduce([mask for _, mask in cannot_use])
    gives = f"a {comparison.measure}"
    if left_out.all():
        raise ValueError(f"None of the {n_parcels} parcels can give {gives}:\n" + _problem_lines(cannot_use, row_labels))
    if left_out.any():
        _warn_data_quality(
            f"pyRealEstate left out {int(left_out.sum())} of {n_parcels} parcels that cannot give {gives}:\n"
            + _problem_lines(cannot_use, row_labels)
            + f"\nThe comparison uses the remaining {int((~left_out).sum())} parcels."
        )
    keep = ~left_out
    denominator, numerator, is_sold, row_labels = denominator[keep], numerator[keep], is_sold[keep], row_labels[keep]

    # Unusual values are kept, but reported.
    unusual = [(f"{above.name} of 0", numerator == 0), (f"negative {above.name}", numerator < 0)]
    if any(mask.any() for _, mask in unusual):
        _warn_data_quality(
            f"pyRealEstate found unusual {above.plural}. These parcels are included in the comparison; please "
            "check that they are correct:\n" + _problem_lines(unusual, row_labels)
        )
    return numerator, denominator, is_sold, row_labels


def _group_summary(values: np.ndarray) -> Tuple[float, float]:
    """The median and mean of one group's values (NaN for an empty group)."""
    if len(values) == 0:
        return math.nan, math.nan
    return float(np.median(values)), float(np.mean(values))


def _compare_sold_unsold(
    values: np.ndarray,
    is_sold: np.ndarray,
    row_labels: np.ndarray,
    center: str,
    significance_level: float,
    comparison: _Comparison,
    unchanged: Optional[np.ndarray] = None,
) -> SoldUnsoldResult:
    """B.1 or B.2 on parcels that have passed the data checks: one value change (or value per unit) each."""
    n_sold = int(is_sold.sum())
    n_unsold = len(is_sold) - n_sold
    sold_median, sold_mean = _group_summary(values[is_sold])
    unsold_median, unsold_mean = _group_summary(values[~is_sold])
    table = {
        "Group": ["Sold", "Unsold"],
        "Parcels": [n_sold, n_unsold],
        comparison.columns[0]: [sold_median, unsold_median],
        comparison.columns[1]: [sold_mean, unsold_mean],
    }
    if unchanged is not None:
        table["Unchanged"] = [float(np.mean(unchanged[is_sold])) if n_sold else math.nan,
                              float(np.mean(unchanged[~is_sold])) if n_unsold else math.nan]
    groups = pd.DataFrame(table)
    parcels = pd.DataFrame({"Sold": is_sold, comparison.parcel_column: values}, index=row_labels)

    def result(difference: float, tolerance: float, statistic: float, share_higher: float, p_value: float,
               p_value_method: str, conclusion: str) -> SoldUnsoldResult:
        return SoldUnsoldResult(
            technique=comparison.technique, center=center, groups=groups, difference=difference,
            tolerance=tolerance, statistic=statistic, share_higher=share_higher, p_value=p_value,
            p_value_method=p_value_method, significance_level=significance_level, conclusion=conclusion,
            parcels=parcels,
        )

    if n_sold == 0 or n_unsold == 0:
        _warn_data_quality(
            f"Appendix {comparison.technique}'s comparison needs both sold and unsold parcels, but the usable "
            f"parcels are {n_sold} sold and {n_unsold} unsold. Its statistics are left empty (NaN)."
        )
        return result(math.nan, math.nan, math.nan, math.nan, math.nan, _MW_NOT_CALCULATED, _SU_NOT_ENOUGH)

    sold_center, unsold_center = (sold_median, unsold_median) if center == "median" else (sold_mean, unsold_mean)
    if comparison.technique == "B.1":
        # Percentage points; 3 points when the sold change is 5% or less (compared to 10 decimals)
        difference = sold_center - unsold_center
        small_change = round(sold_center, 10) <= _B1_SMALL_CHANGE
        tolerance = _B1_SMALL_CHANGE_TOLERANCE if small_change else _B1_TOLERANCE
    elif unsold_center > 0:
        difference = 100 * (sold_center / unsold_center - 1)  # percent of the unsold parcels' value per unit
        tolerance = _B2_TOLERANCE
    else:
        _warn_data_quality(
            f"The unsold parcels' {center} value per unit is {unsold_center:.2f}, 0 or less, so the relative "
            "difference cannot be calculated and the 5% tolerance cannot be applied."
        )
        difference = tolerance = math.nan

    ranks = pd.Series(values, dtype=float).rank(method="average")
    u, p_value, p_value_method, smallest_p_value = _mann_whitney_test(ranks, is_sold)
    share_higher = u / (n_sold * n_unsold)
    _warn_small_mann_whitney_groups(
        groups["Group"], groups["Parcels"], p_value_method, smallest_p_value, significance_level, noun="parcel"
    )

    if p_value_method == _MW_NOT_CALCULATED:  # every value is identical: there is nothing to rank
        _warn_data_quality(
            f"Every {comparison.measure} is the same ({values[0]:.2f}), so the Mann-Whitney test's p-value cannot "
            "be calculated: there is nothing to rank. The groups do not differ."
        )
        conclusion = _SU_NOT_SIGNIFICANT
    elif p_value >= significance_level:
        conclusion = _SU_NOT_SIGNIFICANT
    elif math.isnan(tolerance):
        conclusion = _SU_NO_TOLERANCE
    elif round(abs(difference), 10) > tolerance:  # "greater than" the tolerance, in either direction
        conclusion = comparison.above_tolerance
    else:
        conclusion = _SU_WITHIN
    return result(difference, tolerance, u, share_higher, p_value, p_value_method, conclusion)


def Value_Change_Comparison(
    prior_value: Any,
    current_value: Any,
    sold: Any,
    center: str = "median",
    significance_level: float = 0.05,
) -> SoldUnsoldResult:
    """
    Did the parcels that sold change in value more (or less) than the rest?

    A check for sales chasing from the IAAO Standard on Ratio Studies (2026),
    Appendix B.1, "Comparison of value changes". If sold and unsold parcels
    in a group are valued the same way, their year-to-year percentage changes
    in value should be alike. If the parcels that sold were revalued from
    their sale prices, or only they were revalued, their changes differ. The
    function compares the median change (or the mean) of the sold and unsold
    parcels of one stratum, tests the difference with the Mann-Whitney test,
    and applies the Standard's practical tolerance. It answers Table 6.3's
    "sold and unsold parcels treated equally", which needs the values of
    every parcel, not just the sales.

    Parameters
    ----------
    prior_value : array-like
        Each parcel's value in the earlier year (numpy array, list, or pandas
        Series), e.g. last year's assessed value.
    current_value : array-like
        Each parcel's value in the later year, in the same order.
    sold : array-like
        True for each parcel that sold (the parcels in the ratio study), False
        for the others; 1 / 0 also work. In the same order.
    center : {'median', 'mean'}, default 'median'
        Which center is compared. B.1: "the median is preferred over the
        mean", but when many unsold parcels did not change, "the median value
        change in that group may not be meaningful, in which case the mean
        should be used". Both are always in ``groups`` and the report.
    significance_level : float, default 0.05
        The difference is significant when the Mann-Whitney p-value is below
        this level.

    Returns
    -------
    SoldUnsoldResult
        Each group's parcels, median and mean change, and share unchanged
        (``groups``); the ``difference`` and its ``tolerance``; the
        Mann-Whitney ``statistic``, ``share_higher``, ``p_value`` and
        ``p_value_method``; and the ``conclusion``. ``print(result)`` shows a
        plain-English report.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``sold`` has values other than True / False (1 / 0); if no parcel is
        usable; if ``center`` is not 'median' or 'mean'; or if
        ``significance_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If parcels are left out (a prior value that is missing, infinite, or 0
        or less; a missing or infinite current value; a missing sold flag); if
        current values are 0 or less (kept); if a group has fewer than 5
        parcels or the group sizes make a significant result impossible; if
        there are no sold or no unsold parcels; or if every change is the same.

    See Also
    --------
    Unit_Value_Comparison : B.2, values per unit of sold vs unsold parcels.
    Ratio_Concentration : B.4, the concentration of sale ratios.

    Notes
    -----
    The calculation::

        value change      = 100 x (current value - prior value) / prior value     (percent)
        center            = median (or mean) of each group's value changes
        difference        = sold center - unsold center                           (percentage points)
        tolerance         = 3 points if the sold center is 5% or less, else 5 points
        Mann-Whitney test = the two groups' value changes ranked together (as
                            RealEstateMetrics.Mann_Whitney: exact with 50 parcels
                            or fewer, else the normal approximation)
        conclusion        = "...sales chasing likely" when the test is significant
                            AND |difference| > tolerance; "...within the
                            tolerance" when only the test is significant

    The Standard's rule (B.1): "statistically significant differences in
    valuation changes of sold and unsold properties greater than 5% likely
    indicate sales chasing that warrants remedial action (e.g., changes of 10
    percent for sold properties and 4 percent for unsold properties).
    However, if value changes for sold properties are less than 5%, the
    tolerable difference should be reduced to 3%, e.g., an increase of 5% for
    sales but less than 2% for unsold properties signals probable sales
    chasing." Statistical significance alone is not enough: "In large
    samples, small differences ... can be proven to be statistically
    significant, yet the actual differences may be slight."

    Three readings of that text, stated plainly:

    * The 5% is in **percentage points** (its example: 10% vs 4%).
    * The 3-point tolerance applies when the sold change is **5% or less**:
      the text says "less than 5%", but its own example has sold changes of
      exactly 5% under the 3% rule. (A uniform 5% trend factor puts many
      changes at exactly 5%, so this matters.)
    * The difference counts **in either direction**: in a falling market,
      chasing cuts sold values to their prices, so sold changes are lower.
      The report states the direction.

    **The median or the mean.** When half or more of a group did not change,
    the report notes it. Be aware that the mean can hide selective
    revaluation: in a simulation where 60% of unsold parcels were left
    unchanged and every sold parcel was revalued, comparing medians flagged
    it in 62-100% of samples, comparing means in only 4-28% (the mean
    difference fell under the tolerance). A high share unchanged among unsold
    parcels but not sold ones is itself a sign of selective revaluation.

    **Before comparing**, B.1 says it is "essential to remove properties that
    have undergone a substantive physical change or a change in use over the
    time period", and others that do not belong (e.g. values not set by the
    valuation agency, or changed on appeal). The library cannot see these;
    leave them out first. Compare within one group (a stratum such as a
    neighborhood or property class) per call.

    **Sensitivity.** The tolerance makes this check deliberately coarse: in
    a simulation, chasing 30% of the sales (a 1.6-point difference) was
    essentially never flagged, and chasing every sale (5.5 points) in about
    half to three quarters of samples; honest rolls were rarely flagged.

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Value_Change_Comparison(
    ...     parcels["Prior_Value"], parcels["Current_Value"], parcels["Sold"]
    ... )
    >>> print(result)                     # plain-English report
    >>> result.conclusion
    >>> for name, stratum in parcels.groupby("Neighborhood"):   # one stratum per call
    ...     print(name, sc.Value_Change_Comparison(
    ...         stratum["Prior_Value"], stratum["Current_Value"], stratum["Sold"]).conclusion)
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    center = _check_center(center)
    current, prior, is_sold, row_labels = _parcel_values(prior_value, current_value, sold, _B1)
    change = 100 * (current - prior) / prior  # in this order, so equal changes are equal floats
    return _compare_sold_unsold(
        change, is_sold, row_labels, center, significance_level, _B1, unchanged=current == prior
    )


def Unit_Value_Comparison(
    value: Any,
    units: Any,
    sold: Any,
    center: str = "median",
    significance_level: float = 0.05,
) -> SoldUnsoldResult:
    """
    Are the parcels that sold valued higher (or lower) per unit than the rest?

    A check for sales chasing from the IAAO Standard on Ratio Studies (2026),
    Appendix B.2, "Comparison of unit values": "If sold and unsold parcels
    are not valued equally, median or average unit values (for example, value
    per square foot) could be dissimilar." The function compares the median
    (or mean) value per unit of the sold and unsold parcels of one stratum,
    tests the difference with the Mann-Whitney test, and applies a 5%
    tolerance.

    Parameters
    ----------
    value : array-like
        Each parcel's appraised value (numpy array, list, or pandas Series).
    units : array-like
        Each parcel's units, in the same order, e.g. square feet of living
        area, acres, or front feet.
    sold : array-like
        True for each parcel that sold, False for the others; 1 / 0 also work.
        In the same order.
    center : {'median', 'mean'}, default 'median'
        Which center is compared ("median or average unit values"). Both are
        always in ``groups`` and the report.
    significance_level : float, default 0.05
        The difference is significant when the Mann-Whitney p-value is below
        this level.

    Returns
    -------
    SoldUnsoldResult
        Each group's parcels and median and mean value per unit
        (``groups``); the ``difference`` (percent) and its ``tolerance``; the
        Mann-Whitney ``statistic``, ``share_higher``, ``p_value`` and
        ``p_value_method``; and the ``conclusion``. ``print(result)`` shows a
        plain-English report.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``sold`` has values other than True / False (1 / 0); if no parcel is
        usable; if ``center`` is not 'median' or 'mean'; or if
        ``significance_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If parcels are left out (units that are missing, infinite, or 0 or
        less; a missing or infinite value; a missing sold flag); if values are
        0 or less (kept); if a group has fewer than 5 parcels or the group
        sizes make a significant result impossible; if there are no sold or no
        unsold parcels; if every value per unit is the same; or if the unsold
        parcels' center is 0 or less (no relative difference).

    See Also
    --------
    Value_Change_Comparison : B.1, value changes of sold vs unsold parcels.

    Notes
    -----
    The calculation::

        value per unit    = value / units
        center            = median (or mean) of each group's values per unit
        difference        = 100 x (sold center / unsold center - 1)      (percent)
        tolerance         = 5%
        Mann-Whitney test = the two groups' values per unit ranked together
        conclusion        = "...investigate for sales chasing" when the test is
                            significant AND |difference| > 5%

    B.2 names no tolerance of its own; B.1 says its tolerance "applies to
    other detection techniques discussed below", and only its 5% carries over
    to unit values (as a relative difference, in either direction).

    **Property mix, stated plainly.** Values per unit differ for honest
    reasons too: larger homes have lower values per square foot, so if the
    homes that sold are smaller (or newer, or better located) than the rest,
    the sold parcels' values per unit are higher without any chasing. In a
    simulation, an honest roll whose sold homes were 20% smaller differed by
    +7.1%, significant in every sample. That is why the conclusion says
    "investigate", never "likely": compare like with like (one size range,
    age group or neighborhood per call). The difference can also point
    down: when the recorded size of sold parcels is changed (a subtle form of
    sales chasing named in the Standard's glossary), their values per unit
    fall.

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Unit_Value_Comparison(
    ...     parcels["Appraised_Value"], parcels["Living_Area"], parcels["Sold"]
    ... )
    >>> print(result)                     # plain-English report
    >>> result.difference, result.conclusion
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    center = _check_center(center)
    value, units, is_sold, row_labels = _parcel_values(value, units, sold, _B2)
    return _compare_sold_unsold(value / units, is_sold, row_labels, center, significance_level, _B2)
