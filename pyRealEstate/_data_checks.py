"""
Shared data-quality warning and message helpers.

Used by ``RealEstateMetrics`` (ratio-study statistics) and ``Time_Trending`` (time models) so
that both report data problems the same way: one warning category, ``DataQualityWarning``, and
plain-English messages that name the affected rows.
"""

import warnings
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd


class DataQualityWarning(UserWarning):
    """
    Warning about sales that were left out or that look unusual.

    These warnings can be silenced or caught on their own, for example::

        import warnings
        from pyRealEstate.RealEstateMetrics import DataQualityWarning

        warnings.simplefilter("ignore", DataQualityWarning)

    (``pyRealEstate.Time_Trending.DataQualityWarning`` is the same class.)
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


def _row_labels(inputs: Sequence[Any], length: int) -> np.ndarray:
    """
    Names for the rows in messages: the pandas index of the first input that has
    one (a Series or DataFrame), otherwise positions starting at 0.
    """
    for values in inputs:
        if isinstance(values, (pd.Series, pd.DataFrame)):
            return values.index.to_numpy()
    return np.arange(length)


def _describe_rows(labels: np.ndarray, noun: str = "row") -> str:
    """Format labels for a message, e.g. 'rows 5, 17, 22' or 'month 23'."""
    shown = ", ".join(str(label) for label in labels[:_MAX_ROWS_LISTED])
    if len(labels) > _MAX_ROWS_LISTED:
        shown += f" and {len(labels) - _MAX_ROWS_LISTED} more"
    return (noun if len(labels) == 1 else noun + "s") + " " + shown


def _problem_lines(problems: List[Tuple[str, np.ndarray]], row_labels: np.ndarray) -> str:
    """One line per problem that occurs, e.g. '  - missing sale price: 2 (rows 5, 17)'."""
    return "\n".join(
        f"  - {description}: {int(mask.sum())} ({_describe_rows(row_labels[mask])})"
        for description, mask in problems
        if mask.any()
    )
