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
* B.3, the split-sample technique: ``Split_Sample_Comparison`` (a ratio
  study of the sales up to the date values could be changed against one of
  the sales after it).
* B.4, comparison of observed versus expected distribution of ratios:
  ``Ratio_Concentration``. It needs only a ratio study's sale prices and
  appraised values, so oversight agencies can use it too.
* B.5, mass appraisal techniques: run the ratio-study functions of
  ``RealEstateMetrics`` on unsold parcels, with an independent model's values
  in place of the sale prices (``y``).
* B.6, independent value chasing: ``RealEstateMetrics.Mann_Whitney``, with
  the ratios of independently valued and of sold properties as its two groups.

Argument conventions. For sale ratios (B.3, B.4), as in ``RealEstateMetrics``::

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
    _COD_INTERVAL_MINIMUM_SALES,
    _MW_NOT_CALCULATED,
    _check_fraction,
    _cod_log_variances,
    _cod_rank_a,
    _format_p_value,
    _mann_whitney_test,
    _percent,
    _ratios_with_groups,
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


# The two charts (B.1 / B.2 and B.3) draw the same two horizontal boxes: the
# group under test in blue at y = 2, above the reference group in gray at y = 1.
_TESTED, _REFERENCE = 2, 1


def _draw_two_boxes(ax: "matplotlib.axes.Axes", tested: np.ndarray, reference: np.ndarray) -> None:
    """
    Two horizontal boxes, every value shown (the outliers as small circles):
    the group under test (blue) above the reference group (gray).
    """
    import matplotlib.colors as mcolors  # imported here: slow to import (see RealEstateMetrics)

    style = _PLOT_STYLE
    boxes = ax.boxplot(
        [tested, reference], positions=[_TESTED, _REFERENCE], widths=0.5, patch_artist=True,
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


def _label_above_box(ax: "matplotlib.axes.Axes", text: str, x: float, position: int) -> None:
    """A bold label just above a box, centered on ``x`` (dark for the tested box, softer for the reference)."""
    color = _PLOT_STYLE["text"] if position == _TESTED else _PLOT_STYLE["text_secondary"]
    ax.annotate(text, (x, position + 0.25), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)


def _label_two_boxes(ax: "matplotlib.axes.Axes", tested_label: str, reference_label: str) -> None:
    """Each box's label on the left, and room above the boxes for their labels."""
    ax.set_yticks([_TESTED, _REFERENCE])
    ax.set_yticklabels([tested_label, reference_label], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0.4, 2.75)


def _add_source_note(ax: "matplotlib.axes.Axes", technique: str) -> None:
    """
    The source, right-aligned with the plot, just below the axis label
    (anchored to the label itself, so the two never overlap, whatever the
    font). Only on the library's own figures: on the caller's axes it could
    land on another panel.
    """
    ax.annotate(f"IAAO Standard on Ratio Studies (2026), Appendix {technique}", xy=(1, 0),
                xycoords=("axes fraction", ax.xaxis.label), xytext=(0, -3), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, color=_PLOT_STYLE["muted"])


def _sold_unsold_draw(result: SoldUnsoldResult, ax: "matplotlib.axes.Axes") -> bool:
    """
    Draw the horizontal box plot: sold (blue) above unsold (gray), the
    tolerance shaded around the unsold center, the centers labeled. Returns
    whether the band was drawn (its label sits above the plot, below the
    title).
    """
    import matplotlib.ticker as mticker  # imported here: slow to import (see RealEstateMetrics)

    style = _PLOT_STYLE
    comparison = _COMPARISONS[result.technique]
    b1 = result.technique == "B.1"
    _style_axes(ax, grid_axis="x")

    # Every parcel, sold above unsold
    measure = result.parcels[comparison.parcel_column].to_numpy()
    sold = result.parcels["Sold"].to_numpy()
    _draw_two_boxes(ax, measure[sold], measure[~sold])

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
    for position, center, marker_color in (
        (_TESTED, sold_center, style["blue_dark"]),
        (_REFERENCE, unsold_center, style["context_dark"]),
    ):
        if result.center == "mean":
            ax.scatter([center], [position], marker="D", s=46, color=marker_color, edgecolor=style["surface"],
                       linewidth=1.5, zorder=4)
        _label_above_box(ax, f"{result.center} {_sold_unsold_value_text(result, center)}", center, position)

    # Each group's parcels (and, for B.1, share unchanged) beside its box
    labels = []
    for row in result.groups.itertuples(index=False):
        label = f"{row.Group}\n{row.Parcels:,} parcel{'' if row.Parcels == 1 else 's'}"
        if b1:
            label += f"\n{row.Unchanged:.1%} unchanged"
        labels.append(label)
    _label_two_boxes(ax, *labels)
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
    if own_figure:
        _add_source_note(ax, result.technique)


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


# ---------------------------------------------------------------------------
# B.3: the split-sample technique
# ---------------------------------------------------------------------------

_B3_LEVEL_TOLERANCE = 5.0  # points: B.1's 5% tolerance "applies to other detection techniques" (user decision)
_B3_COD_TOLERANCE = 20.0  # percent lower: our judgment; the Standard gives no number (user decision, F-159)

# The conclusions returned by Split_Sample_Comparison
_B3_INDICATED = "Sales chasing indicated: use the after-cutoff study's {statistic}"
_B3_WITHIN = "Better before the cutoff, within the tolerance"
_B3_NOT_BETTER = "Significant difference, not better before the cutoff"
_B3_NOT_SIGNIFICANT = "No significant difference"
_B3_NONE_INDICATED = "No sales chasing indicated"
_B3_NOT_ENOUGH = "Not enough sales (both studies are needed)"
_B3_COD_TOO_FEW = f"Not calculated (a study has fewer than {_COD_INTERVAL_MINIMUM_SALES} sales)"
_B3_COD_IDENTICAL = "Not calculated (every sale ratio in a study is the same)"
_B3_COD_LOG = "Not calculated (a sale ratio of 0 or less)"


@dataclass(frozen=True, eq=False, repr=False)
class SplitSampleResult:
    """
    The result of ``Split_Sample_Comparison``: is the ratio study of the sales
    the valuation agency could have seen suspiciously better than the study of
    the sales it could not? (IAAO Standard on Ratio Studies 2026, Appendix
    B.3.)

    ``print(result)`` shows a plain-English report. The values are read-only.

    Attributes
    ----------
    cutoff_date : pandas.Timestamp
        The cutoff: sales on or before it form the first study, sales after
        it the second.
    sales_before, sales_after : int
        The number of usable sales in each study.
    median_before, median_after : float
        Each study's median sale ratio.
    level_difference : float
        How much closer to 1.00 the first study's median ratio is, in points
        (100 x (|median_after - 1| - |median_before - 1|)); negative when the
        second study's is closer.
    level_statistic, level_p_value : float
        The Mann-Whitney U of the first study and the two-sided p-value of
        the test that the two studies' ratios sit at the same level.
    level_p_value_method : str
        ``'exact'``, ``'normal approximation'`` or ``'not calculated'``.
    level_conclusion : str
        ``"Sales chasing indicated: use the after-cutoff study's median
        ratio"``, ``'Better before the cutoff, within the tolerance'``,
        ``'Significant difference, not better before the cutoff'``, ``'No
        significant difference'``, or ``'Not enough sales (both studies are
        needed)'``.
    cod_before, cod_after : float
        Each study's COD, in percent.
    cod_difference : float
        How much the first study's COD differs from the second's, in percent
        of the second's (100 x (cod_before / cod_after - 1)); negative when
        it is lower.
    cod_p_value : float
        The one-sided p-value of the test that the first study's COD is
        lower (a z-test on log COD; see ``Split_Sample_Comparison``'s Notes).
        NaN when it cannot be calculated.
    cod_conclusion : str
        As ``level_conclusion`` (without "not better": the test is
        one-sided), or ``'Not calculated (...)'`` with the reason.
    level_tolerance, cod_tolerance : float
        The practical tolerances used: points for the level, percent for the
        COD.
    significance_level : float
        The significance level used for the conclusions, e.g. 0.05.
    conclusion : str
        ``"Sales chasing indicated (<statistics>): use the after-cutoff
        study's <statistics>"``, ``'No sales chasing indicated'``, or ``'Not
        enough sales (both studies are needed)'``.
    sales : pandas.DataFrame
        One row per usable sale, labeled like the input's rows: ``Sale_Date``,
        ``Sale_Ratio``, and ``After`` (True for the second study).

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Split_Sample_Comparison(
    ...     sales["Adjusted_Sale_Price"], sales["Appraised_Value"], sales["Sale_Date"], cutoff_date="2026-01-01"
    ... )
    >>> result.conclusion
    >>> result.median_before, result.median_after, result.cod_before, result.cod_after
    >>> print(result)                  # plain-English report
    """

    cutoff_date: pd.Timestamp
    sales_before: int
    sales_after: int
    median_before: float
    median_after: float
    level_difference: float
    level_statistic: float
    level_p_value: float
    level_p_value_method: str
    level_conclusion: str
    cod_before: float
    cod_after: float
    cod_difference: float
    cod_p_value: float
    cod_conclusion: str
    level_tolerance: float
    cod_tolerance: float
    significance_level: float
    conclusion: str
    sales: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"SplitSampleResult(median_before={self.median_before:.4f}, median_after={self.median_after:.4f}, "
            f"cod_before={self.cod_before:.2f}, cod_after={self.cod_after:.2f}, conclusion={self.conclusion!r})"
        )

    def __str__(self) -> str:
        """A plain-English report of the result (ASCII only, so it prints anywhere)."""
        lines = [
            "Split-sample comparison: a check for sales chasing",
            "(IAAO Standard on Ratio Studies, 2026: Appendix B.3)",
            f"Cutoff {self.cutoff_date:%Y-%m-%d}: {self.sales_before} sales on or before it, "
            f"{self.sales_after} after it",
        ]
        if self.conclusion == _B3_NOT_ENOUGH:
            lines.append(f"Conclusion: {self.conclusion}")
            return "\n".join(lines)
        lines += [
            self._level_text(),
            f"  {self.level_conclusion}",
            self._cod_text(),
            f"  {self.cod_conclusion}",
            f"Conclusion at the {_percent(self.significance_level)} significance level: {self.conclusion}",
            "Sale prices must be adjusted for time (B.3): a market trend left in them looks like a difference "
            "between the studies.",
        ]
        return "\n".join(lines)

    def _level_text(self) -> str:
        """The median ratio line of the report."""
        difference = round(self.level_difference, 10)
        if difference > 0:
            closer = f"{difference:.2f} points closer to 1.00 before (tolerance {self.level_tolerance:g})"
        elif difference < 0:
            closer = f"{-difference:.2f} points closer to 1.00 after"
        else:
            closer = "equally close to 1.00"
        if math.isnan(self.level_p_value):
            test = "Mann-Whitney test not calculated (every sale ratio is the same)"
        else:
            test = f"Mann-Whitney p {_format_p_value(self.level_p_value)} ({self.level_p_value_method})"
        return f"Median ratio: {self.median_before:.4f} before, {self.median_after:.4f} after; {closer}; {test}"

    def _cod_text(self) -> str:
        """The COD line of the report."""
        difference = round(self.cod_difference, 10)
        if math.isnan(difference):
            change = "no relative difference (the after-cutoff COD is 0)"
        elif difference < 0:
            change = f"{-difference:.1f}% lower before (tolerance {self.cod_tolerance:g}%)"
        elif difference > 0:
            change = f"{difference:.1f}% higher before"
        else:
            change = "the same"
        if math.isnan(self.cod_p_value):
            test = "z-test on log COD not calculated"
        else:
            test = f"one-sided z-test on log COD p {_format_p_value(self.cod_p_value)}"
        return f"COD: {self.cod_before:.2f} before, {self.cod_after:.2f} after; {change}; {test}"

    def plot(
        self,
        title: Optional[str] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        show: bool = True,
    ) -> None:
        """
        Chart the two ratio studies of Appendix B.3 of the IAAO Standard on
        Ratio Studies (2026): a box plot of the sale ratios on or before the
        cutoff and after it.

        The first study (on or before the cutoff, the sales the valuation
        agency could have seen) is the blue box, the group under test; the
        second (after the cutoff) is the gray box below it, the reference.
        Every sale is shown, the outliers as small circles. A line marks 1.00
        (market value), since the level test asks which median is closer to
        it; both medians are labeled, and each study's sales and COD are
        beside its box (a narrower box is a lower COD). The title states the
        verdict; the subtitle gives the cutoff and both tests.

        Parameters
        ----------
        title : str, optional
            Your own title instead of the default ("Split-sample comparison:
            <verdict>"). The subtitle then starts with the verdict, so the
            chart still states it.
        ax : matplotlib.axes.Axes, optional
            Draw on these axes (e.g. one panel per stratum) instead of a new
            figure, leaving your figure's colors alone.
        show : bool, default True
            Display the chart. Use False to keep customizing it or to save
            it, for example with ``plt.savefig("split_sample.png", dpi=300)``.

        Returns
        -------
        None
            The chart is drawn on a new figure, or on ``ax``.
            ``plt.gcf()`` returns the figure.

        Notes
        -----
        Chasing shows as a first (blue) study that is narrower (a lower COD)
        and centered nearer 1.00 than the second. With sales on only one side
        of the cutoff there is nothing to compare, so the chart shows only its
        title and the reason.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from pyRealEstate import Sales_Chasing as sc
        >>> result = sc.Split_Sample_Comparison(
        ...     sales["Adjusted_Sale_Price"], sales["Appraised_Value"], sales["Sale_Date"], cutoff_date="2026-01-01"
        ... )
        >>> result.plot()
        >>> result.plot(title="Residential, 2026 roll", show=False)
        >>> plt.savefig("split_sample.png", dpi=300)
        """
        import matplotlib.pyplot as plt  # imported here: slow to import (see RealEstateMetrics)

        if self.conclusion == _B3_NOT_ENOUGH:
            _split_sample_plot_not_enough(self, title, ax)
        else:
            _split_sample_plot(self, title, ax)
        if show:
            plt.show()


def _split_sample_verdict(result: SplitSampleResult) -> str:
    """The verdict for the chart's title: 'sales chasing indicated (median ratio, COD)', 'no sales chasing ...'."""
    verdict = result.conclusion.split(":")[0]  # drops ": use the after-cutoff study's ..."
    return verdict[0].lower() + verdict[1:]


def _split_sample_facts(result: SplitSampleResult) -> List[str]:
    """
    The subtitle's parts: the cutoff and the two comparisons. The medians and
    CODs themselves are labeled on the chart, so they are not repeated, and
    each part fits a stacked panel (the subtitle wraps only between parts).
    """
    difference = round(result.level_difference, 10)
    if difference > 0:
        closer = f"{difference:.2f} points closer to 1.00 before (tolerance {result.level_tolerance:g})"
    elif difference < 0:
        closer = f"{-difference:.2f} points closer to 1.00 after"
    else:
        closer = "equally close to 1.00"
    if math.isnan(result.level_p_value):
        level_test = "Mann-Whitney test not calculated"
    else:
        level_test = f"p {_format_p_value(result.level_p_value)}"

    change = round(result.cod_difference, 10)
    if math.isnan(change):
        cod_change = "no relative difference"
    elif change < 0:
        cod_change = f"{-change:.1f}% lower before (tolerance {result.cod_tolerance:g}%)"
    elif change > 0:
        cod_change = f"{change:.1f}% higher before"
    else:
        cod_change = "the same"
    if math.isnan(result.cod_p_value):
        cod_test = f"{result.cod_conclusion[0].lower()}{result.cod_conclusion[1:]}"
    else:
        cod_test = f"p {_format_p_value(result.cod_p_value)}"

    return [
        f"Cutoff {result.cutoff_date:%Y-%m-%d}",
        f"Median ratio: {closer}, {level_test}",
        f"COD: {cod_change}, {cod_test}",
    ]


def _split_sample_plot(
    result: SplitSampleResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]
) -> None:
    """``SplitSampleResult.plot()`` when both studies have sales."""
    style = _PLOT_STYLE
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 3.9))
    _style_axes(ax, grid_axis="x")

    # Every sale: the first study (before the cutoff) above the second
    ratios = result.sales["Sale_Ratio"].to_numpy()
    after = result.sales["After"].to_numpy()
    _draw_two_boxes(ax, ratios[~after], ratios[after])

    # 1.00, market value: the level test asks which median is closer to it
    ax.axvline(1.0, color=style["axis"], linewidth=1, zorder=1)
    ax.text(1.0, 1.0, "market value (1.00)", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8.5, color=style["muted"])
    _label_above_box(ax, f"median {result.median_before:.4f}", result.median_before, _TESTED)
    _label_above_box(ax, f"median {result.median_after:.4f}", result.median_after, _REFERENCE)

    _label_two_boxes(
        ax,
        f"Before the cutoff\n{result.sales_before:,} sale{'' if result.sales_before == 1 else 's'}\n"
        f"COD {result.cod_before:.2f}",
        f"After the cutoff\n{result.sales_after:,} sale{'' if result.sales_after == 1 else 's'}\n"
        f"COD {result.cod_after:.2f}",
    )
    ax.set_xlabel("Sale ratio (appraised value ÷ time-adjusted sale price)", color=style["text_secondary"],
                  fontsize=10)

    verdict = _split_sample_verdict(result)
    facts = _split_sample_facts(result)
    if title is None:
        title = f"Split-sample comparison: {verdict}"
    else:
        facts = [verdict[0].upper() + verdict[1:]] + facts
    title, subtitle = _fit_title(ax, own_figure, title, facts)
    _add_title(ax, title, subtitle, offset=4 + 14)  # 14: room for the market-value label above the plot
    if own_figure:
        _add_source_note(ax, "B.3")


def _split_sample_plot_not_enough(
    result: SplitSampleResult, title: Optional[str], ax: Optional["matplotlib.axes.Axes"]
) -> None:
    """With sales on only one side of the cutoff there is nothing to compare: only the title and the reason."""
    own_figure = ax is None
    if own_figure:
        ax = _new_axes(figsize=(9, 1.0))
    ax.axis("off")
    reason = (f"{result.sales_before:,} sales on or before {result.cutoff_date:%Y-%m-%d} and "
              f"{result.sales_after:,} after it; the comparison needs both")
    if title is None:
        title, facts = "Split-sample comparison: not enough sales", [reason]
    else:
        facts = [result.conclusion, reason]
    title, subtitle = _fit_title(ax, own_figure, title, facts)
    _add_title(ax, title, subtitle)


def _check_tolerances(level_tolerance: Any, cod_tolerance: Any) -> None:
    """Raise a ValueError unless the level tolerance is 0 or more points and the COD's 0 to under 100 percent."""
    for value, name, low, high, example in (
        (level_tolerance, "level_tolerance", 0.0, math.inf, "a number of points, 0 or more (for example 5.0)"),
        (cod_tolerance, "cod_tolerance", 0.0, 100.0, "a percentage from 0 up to 100, not 100 itself (for example 20.0)"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or not low <= value < high
        ):
            raise ValueError(f"{name} must be {example}, not {value!r}.")


def _check_cutoff(cutoff_date: Any) -> pd.Timestamp:
    """The cutoff as a Timestamp (time zone dropped); anything that is not a date raises a ValueError."""
    message = (
        "cutoff_date must be a date, the last date the valuation agency could change values (for example "
        f"'2026-01-01'), not {cutoff_date!r}."
    )
    if isinstance(cutoff_date, (bool, numbers.Number)):  # a number would be read as nanoseconds since 1970
        raise ValueError(message)
    try:
        cutoff = pd.Timestamp(cutoff_date)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if pd.isna(cutoff):
        raise ValueError(message)
    return cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff


def _sale_dates(y: Any, sale_date: Any) -> np.ndarray:
    """
    The sale dates as datetime64, paired with the sales by position; a date
    that is missing or cannot be read is NaT (the sale is then left out, with
    a warning). Numbers, a different length, or no usable date at all raise a
    ValueError.
    """
    raw = pd.Series(sale_date).to_numpy()
    if len(raw) != len(y):
        raise ValueError(
            f"There are {len(y)} sale prices (y) but {len(raw)} sale dates; each sale needs exactly one sale date."
        )
    if pd.api.types.is_numeric_dtype(raw):
        raise ValueError("sale_date must be dates (for example '2026-03-15', or datetime values), not numbers.")
    dates = pd.to_datetime(pd.Series(raw), errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    if len(dates) and dates.isna().all():
        raise ValueError(
            f"None of the {len(dates)} sales has a usable sale date: sale_date must hold dates, for example "
            "'2026-03-15'."
        )
    return dates.to_numpy()


def _cod_of(ratios: np.ndarray) -> float:
    """The COD in percent, calculated as RealEstateMetrics.COD does."""
    median = np.median(ratios)
    return float(100 * ((np.sum(np.abs(ratios - median)) / len(ratios)) / median))


def _log_cod_and_variance(ratios: np.ndarray, when: str) -> Tuple[float, float, Optional[str]]:
    """
    One study's log COD and its variance, from Appendix D.6's pieces, or NaN
    with the reason (a conclusion) and a warning when they cannot be
    calculated. ``when`` is "before the cutoff" or "after the cutoff".
    """
    ratios = np.sort(ratios)
    n = len(ratios)
    median = float(np.median(ratios))
    tau = float(np.mean(np.abs(ratios - median)))
    if tau == 0:
        _warn_data_quality(
            f"Every sale ratio {when} is the same ({ratios[0]:.4f}), so its COD is 0 and the split-sample COD "
            "test cannot be calculated. Identical ratios are a strong sign of sales chasing: see "
            "Ratio_Concentration (Appendix B.4)."
        )
        return math.nan, math.nan, _B3_COD_IDENTICAL
    rank_a = _cod_rank_a(n)
    if ratios[rank_a - 1] <= 0:
        _warn_data_quality(
            f"The sale ratio ranked {rank_a} of {n} {when} is {ratios[rank_a - 1]:.4f}, 0 or less, so the "
            "split-sample COD test cannot be calculated (Appendix D.6 takes its logarithm)."
        )
        return math.nan, math.nan, _B3_COD_LOG
    var_log_median, var_log_tau, covariance = _cod_log_variances(ratios, median, tau)
    # log COD with the bias correction of step 18 (tau x n / (n - 1)), so a smaller study does not look better
    log_cod = math.log(n / (n - 1) * tau) - math.log(median)
    return log_cod, var_log_median + var_log_tau - 2 * covariance, None


def _split_sample_cod_test(before: np.ndarray, after: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    The one-sided p-value of "the first study's COD is lower", or NaN with
    the reason. A z-test on log COD: z = (log COD_before - log COD_after) /
    sqrt(var_before + var_after), p = P(Z <= z).
    """
    if min(len(before), len(after)) < _COD_INTERVAL_MINIMUM_SALES:
        _warn_data_quality(
            f"The split-sample COD test needs at least {_COD_INTERVAL_MINIMUM_SALES} sales in each study "
            f"(Appendix D.6), but there are {len(before)} before the cutoff and {len(after)} after it. The COD "
            "part is not calculated; the level test still runs."
        )
        return math.nan, _B3_COD_TOO_FEW
    log_before, var_before, reason = _log_cod_and_variance(before, "before the cutoff")
    if reason is None:
        log_after, var_after, reason = _log_cod_and_variance(after, "after the cutoff")
    if reason is not None:
        return math.nan, reason
    z = (log_before - log_after) / math.sqrt(var_before + var_after)
    return 0.5 * math.erfc(-z / math.sqrt(2)), None  # P(Z <= z) for a standard normal Z


def Split_Sample_Comparison(
    y: Any,
    x: Any,
    sale_date: Any,
    cutoff_date: Any,
    level_tolerance: float = _B3_LEVEL_TOLERANCE,
    cod_tolerance: float = _B3_COD_TOLERANCE,
    significance_level: float = 0.05,
) -> SplitSampleResult:
    """
    Is the ratio study of the sales the assessor could see suspiciously
    better than the study of the sales it could not?

    A check for sales chasing from the IAAO Standard on Ratio Studies (2026),
    Appendix B.3, "Split sample technique": "two ratio studies are
    performed, one using sales that occurred before the point at which sales
    would have been known by the valuation agency and one using sales that
    occurred after that point, both adjusted for date of sale as
    appropriate ... the results of the two studies should be similar. Sales
    chasing is indicated if a given statistic in the first study is
    statistically significantly better than in the second study and that the
    difference is practically meaningful. In such a case, the given statistic
    in the second study is still valid but the statistic in the first study
    should be rejected."

    The function compares the two studies' level (the median ratio) and
    uniformity (the COD), the two statistics chasing distorts: chased ratios
    "appear more uniform than they are", and the level "likely closer to
    market value" than it is (Appendix B).

    Parameters
    ----------
    y : array-like
        Sale prices **adjusted for time** to the appraisal date (numpy array,
        list, or pandas Series), e.g. with a ``Time_Trending`` model's
        adjustment factors. B.3 requires it: a market trend left in the
        prices shifts one study's level and looks like a difference.
    x : array-like
        Appraised values, in the same order as ``y``.
    sale_date : array-like
        Each sale's date, in the same order: strings such as '2026-03-15',
        datetimes, dates, or a pandas datetime column.
    cutoff_date : date-like
        The last date at which the valuation agency could change values other
        than through appeal (B.3's "practical dividing point"), e.g.
        '2026-01-01'. Sales on or before it form the first study; sales after
        it, the second.
    level_tolerance : float, default 5.0
        How much closer to 1.00, in points, the first study's median ratio
        must be for the difference to count as practically meaningful (more
        than this). The default is B.1's 5% tolerance, which "applies to other
        detection techniques".
    cod_tolerance : float, default 20.0
        How much lower, in percent, the first study's COD must be (at least
        this). The Standard gives no number; 20% is the library's judgment
        (see Notes).
    significance_level : float, default 0.05
        A difference is significant when its p-value is below this level.

    Returns
    -------
    SplitSampleResult
        Both studies' sales, median ratios and CODs, the two tests, a
        conclusion for each statistic and overall, and each sale's date,
        ratio and study (``sales``). ``print(result)`` shows a plain-English
        report.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or are not numbers; if
        ``sale_date`` holds numbers or no usable date; if ``cutoff_date`` is
        not a date; if a tolerance is out of range; or if
        ``significance_level`` is not between 0 and 1.

    Warns
    -----
    DataQualityWarning
        If sales are left out (see "Data checks" in ``RealEstateMetrics``),
        including sales without a usable sale date; if a study has fewer than
        5 sales or the study sizes make a significant level difference
        impossible; if one side of the cutoff has no sales; or if the COD
        test cannot be calculated (fewer than 5 sales in a study, identical
        ratios, or a ratio of 0 or less where its logarithm is needed).

    See Also
    --------
    Ratio_Concentration : B.4, the concentration of sale ratios.
    Value_Change_Comparison : B.1, value changes of sold vs unsold parcels.
    Time_Trending.SPPSF_Polynomial_Time_Model : adjustment factors for the
        sale prices.

    Notes
    -----
    **The Standard names no test, statistic or tolerance for B.3.** The
    choices below are the library's (decided with its maintainer and
    measured on simulated split samples):

    * **The level** is better when the median ratio is closer to 1.00
      (market value). The test is the Mann-Whitney test on the two studies'
      ratios (two-sided; exact with 50 sales or fewer, as
      ``RealEstateMetrics.Mann_Whitney``). Chasing pulls the first study's
      median toward 1.00 (in a simulation at a level of 0.92, 40% of sales
      chased: 0.99 vs 0.92); at a level of 1.00 it does not move the median,
      which is why the COD is compared too.
    * **The COD** is better when lower. The test is one-sided, a z-test on
      log COD built from Appendix D.6's own pieces (Bonett and Seier 2006)::

          log COD   = log(n / (n - 1) x tau) - log(median)     (tau: mean absolute deviation from the median)
          var       = var_log_median + var_log_tau - 2 x covariance   (D.6, steps 8-14)
          z         = (log COD_before - log COD_after) / sqrt(var_before + var_after)
          p-value   = P(Z <= z)

      This is an extension of D.6's one-study interval, not a formula the
      Standard prints. With no real difference it flagged 4.0-5.5% of
      simulated samples at a nominal 5% (normal, log-normal, skewed and
      heavy-tailed ratios, 30 to 300 sales), and it had about twice the power
      of checking whether the two studies' 90% D.6 intervals overlap.
    * **Practically meaningful:** the level must be more than
      ``level_tolerance`` points closer to 1.00, the COD at least
      ``cod_tolerance`` percent lower. Without a tolerance, large samples
      flag trivial differences (the Standard: "small differences ... can be
      proven to be statistically significant, yet the actual differences
      may be slight"); with 20%, honest simulated samples of 300 and 1,000
      sales per study were never flagged, while 40% of sales chased was
      caught in 94-100% of them. A relative COD tolerance suits every
      property type; 20% chased lowers the COD by about 20%, so it is caught
      about half the time.

    The conclusion for each statistic: "Sales chasing indicated: use the
    after-cutoff study's <statistic>" when its test is significant and the
    difference is beyond the tolerance (B.3: "the given statistic in the
    second study is still valid but the statistic in the first study should
    be rejected"); "Better before the cutoff, within the tolerance" when
    only the test is; "Significant difference, not better before the
    cutoff" when the levels differ but the second is closer to 1.00; or "No
    significant difference".

    The two studies must be different sales: each sale belongs to one side
    of the cutoff. A sale dated on the cutoff is in the first study.

    Examples
    --------
    >>> from pyRealEstate import Sales_Chasing as sc
    >>> result = sc.Split_Sample_Comparison(
    ...     sales["Adjusted_Sale_Price"], sales["Appraised_Value"], sales["Sale_Date"], cutoff_date="2026-01-01"
    ... )
    >>> print(result)                     # plain-English report
    >>> result.conclusion
    >>> result.level_conclusion, result.cod_conclusion
    """
    _check_fraction(significance_level, "significance_level", "for example 0.05 for 5%")
    _check_tolerances(level_tolerance, cod_tolerance)
    cutoff = _check_cutoff(cutoff_date)
    dates = _sale_dates(y, sale_date)
    ratio, dates, row_labels = _ratios_with_groups(
        y, x, dates, "split-sample", label_noun="usable sale date", problem="missing or unreadable sale date",
        return_rows=True,
    )
    after = dates > cutoff.to_datetime64()
    before_ratios, after_ratios = ratio[~after], ratio[after]
    n_before, n_after = len(before_ratios), len(after_ratios)
    sales = pd.DataFrame({"Sale_Date": dates, "Sale_Ratio": ratio, "After": after}, index=row_labels)

    def result(**values: Any) -> SplitSampleResult:
        return SplitSampleResult(
            cutoff_date=cutoff, sales_before=n_before, sales_after=n_after, level_tolerance=float(level_tolerance),
            cod_tolerance=float(cod_tolerance), significance_level=significance_level, sales=sales, **values,
        )

    if n_before == 0 or n_after == 0:
        _warn_data_quality(
            f"The split-sample test needs sales on both sides of the cutoff ({cutoff:%Y-%m-%d}), but there are "
            f"{n_before} sales on or before it and {n_after} after it. Its statistics are left empty (NaN)."
        )
        empty = {"median_before": math.nan, "median_after": math.nan, "level_difference": math.nan,
                 "level_statistic": math.nan, "level_p_value": math.nan, "level_p_value_method": _MW_NOT_CALCULATED,
                 "cod_before": math.nan, "cod_after": math.nan, "cod_difference": math.nan, "cod_p_value": math.nan}
        return result(**empty, level_conclusion=_B3_NOT_ENOUGH, cod_conclusion=_B3_NOT_ENOUGH,
                      conclusion=_B3_NOT_ENOUGH)

    # The level: closer to 1.00 is better; the Mann-Whitney test on the two studies' ratios
    median_before, median_after = float(np.median(before_ratios)), float(np.median(after_ratios))
    level_difference = 100 * (abs(median_after - 1) - abs(median_before - 1))
    ranks = pd.Series(ratio, dtype=float).rank(method="average")
    u, level_p_value, level_method, smallest_p_value = _mann_whitney_test(ranks, ~after)
    _warn_small_mann_whitney_groups(
        ["Before the cutoff", "After the cutoff"], [n_before, n_after], level_method, smallest_p_value,
        significance_level,
    )
    if level_method == _MW_NOT_CALCULATED:
        _warn_data_quality(
            f"Every sale ratio is the same ({ratio[0]:.4f}), so the split-sample level test cannot be calculated: "
            "there is nothing to rank. Identical ratios are a strong sign of sales chasing: see "
            "Ratio_Concentration (Appendix B.4)."
        )
        level_conclusion = _B3_NOT_SIGNIFICANT
    elif level_p_value >= significance_level:
        level_conclusion = _B3_NOT_SIGNIFICANT
    elif round(level_difference, 10) <= 0:
        level_conclusion = _B3_NOT_BETTER
    elif round(level_difference, 10) > round(level_tolerance, 10):  # "more than" the tolerance
        level_conclusion = _B3_INDICATED.format(statistic="median ratio")
    else:
        level_conclusion = _B3_WITHIN

    # The COD: lower is better; the one-sided z-test on log COD
    cod_before, cod_after = _cod_of(before_ratios), _cod_of(after_ratios)
    cod_difference = 100 * (cod_before / cod_after - 1) if cod_after > 0 else math.nan
    cod_p_value, cod_reason = _split_sample_cod_test(before_ratios, after_ratios)
    if cod_reason is not None:
        cod_conclusion = cod_reason
    elif cod_p_value >= significance_level:
        cod_conclusion = _B3_NOT_SIGNIFICANT
    elif round(-cod_difference, 10) >= round(cod_tolerance, 10):  # "at least" the tolerance
        cod_conclusion = _B3_INDICATED.format(statistic="COD")
    else:
        cod_conclusion = _B3_WITHIN

    indicated = [name for name, conclusion in (("median ratio", level_conclusion), ("COD", cod_conclusion))
                 if conclusion == _B3_INDICATED.format(statistic=name)]
    if indicated:
        conclusion = (f"Sales chasing indicated ({', '.join(indicated)}): use the after-cutoff study's "
                      f"{' and '.join(indicated)}")
    else:
        conclusion = _B3_NONE_INDICATED
    return result(
        median_before=median_before, median_after=median_after, level_difference=level_difference,
        level_statistic=u, level_p_value=level_p_value, level_p_value_method=level_method,
        level_conclusion=level_conclusion, cod_before=cod_before, cod_after=cod_after,
        cod_difference=cod_difference, cod_p_value=cod_p_value, cod_conclusion=cod_conclusion,
        conclusion=conclusion,
    )
