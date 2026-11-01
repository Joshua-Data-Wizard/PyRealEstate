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

* B.4, comparison of observed versus expected distribution of ratios:
  ``Ratio_Concentration``. It needs only a ratio study's sale prices and
  appraised values, so oversight agencies can use it too.
* B.5, mass appraisal techniques: run the ratio-study functions of
  ``RealEstateMetrics`` on unsold parcels, with an independent model's values
  in place of the sale prices (``y``).
* B.6, independent value chasing: ``RealEstateMetrics.Mann_Whitney``, with
  the ratios of independently valued and of sold properties as its two groups.
* B.1 (value changes), B.2 (unit values) and B.3 (split sample) are not in
  the library yet.

Argument convention, as in ``RealEstateMetrics``::

    y : sale prices
    x : appraised / model-predicted values
    sale ratio = appraised value / sale price

The sales are checked as in ``RealEstateMetrics`` (see "Data checks" there):
sales that cannot form a ratio are left out, appraised values of 0 or less
are kept, each with a ``DataQualityWarning``, and impossible inputs raise a
``ValueError``.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from ._data_checks import DataQualityWarning, _warn_data_quality  # DataQualityWarning: so users can import it here
from .RealEstateMetrics import _sales_pairs

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
