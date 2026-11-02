"""
Data-preparation helpers for mass appraisal datasets.

Each function takes a pandas DataFrame (for example, one row per sale or
parcel) and reports on its columns: which have missing values, which have
(almost) no variation, and which look like row IDs. Nothing is changed or
removed; the results help decide which columns to clean or leave out before
modeling.

Missing values (NaN, None, NaT) are never counted as a distinct value.
"""

from typing import Any, Hashable, List

import pandas as pd


def _check_dataframe(dataf: Any) -> None:
    """Raise a clear error if ``dataf`` is not a pandas DataFrame."""
    if not isinstance(dataf, pd.DataFrame):
        raise TypeError(
            f"dataf must be a pandas DataFrame, not {type(dataf).__name__}."
        )


def Pandas_Missing_Data(
    dataf: pd.DataFrame,
    Percent_Missing=-1
) -> pd.DataFrame:
    """
    Count the missing values in each column.

    Parameters
    ----------
    dataf : pandas.DataFrame
        The data, for example one row per sale or parcel.
    Percent_Missing : float, default -1
        Only columns whose percentage of missing values is greater than this
        are returned. The default, -1, returns every column; 0 returns only
        the columns that have at least one missing value.

    Returns
    -------
    pandas.DataFrame
        One row per column of ``dataf`` (indexed by column name) with the
        columns ``column_name``, ``missing_count`` and ``percent_missing``
        (0 to 100). A DataFrame with no rows gives an empty table.

    Raises
    ------
    TypeError
        If ``dataf`` is not a pandas DataFrame.

    Examples
    --------
    >>> from pyRealEstate import Pre_Processing as pp
    >>> pp.Pandas_Missing_Data(sales, Percent_Missing=0)  # columns with any missing values
    """
    _check_dataframe(dataf)

    missing_count = dataf.isnull().sum(axis=0)
    percent_missing = missing_count * 100 / len(dataf)
    missing_value_df = pd.DataFrame(dict(
        column_name=dataf.columns,
        missing_count=missing_count,
        percent_missing=percent_missing
    ))
    return (
        missing_value_df[missing_value_df['percent_missing'] > Percent_Missing]
    )


def Constant_In_Data(dataf: pd.DataFrame, n_unique=1) -> List[Hashable]:
    """
    Find columns with very few distinct values.

    A column with a single distinct value (a constant column) carries no
    information for a model, and a column with only a few distinct values may
    be worth reviewing. A completely empty column (every value missing, so 0
    distinct values) carries no information either, so it is always included.

    Parameters
    ----------
    dataf : pandas.DataFrame
        The data, for example one row per sale or parcel.
    n_unique : int, default 1
        Return the columns with at most this many distinct (non-missing)
        values.

    Returns
    -------
    list
        The names of the matching columns, in the order they appear in
        ``dataf``. With no rows, every column counts as empty.

    Raises
    ------
    TypeError
        If ``dataf`` is not a pandas DataFrame.

    Examples
    --------
    >>> from pyRealEstate import Pre_Processing as pp
    >>> pp.Constant_In_Data(sales)               # constant or empty columns
    >>> pp.Constant_In_Data(sales, n_unique=2)   # ... or with only 2 values
    """
    _check_dataframe(dataf)

    distinct_values = dataf.nunique()
    return distinct_values[distinct_values <= n_unique].index.tolist()


def ID_in_Data(dataf: pd.DataFrame) -> List[Hashable]:
    """
    Find columns that look like row IDs: every row has a different value.

    Such columns (for example parcel or sale numbers) identify rows rather
    than describe them, so they are usually left out of a model.

    Parameters
    ----------
    dataf : pandas.DataFrame
        The data, for example one row per sale or parcel.

    Returns
    -------
    list
        The names of the columns whose values are all different, in the order
        they appear in ``dataf``. A column with any missing value is not
        included, and a DataFrame with no rows has no ID columns.

    Raises
    ------
    TypeError
        If ``dataf`` is not a pandas DataFrame.

    Examples
    --------
    >>> from pyRealEstate import Pre_Processing as pp
    >>> pp.ID_in_Data(sales)
    """
    _check_dataframe(dataf)
    if len(dataf) == 0:
        return []

    distinct_values = dataf.nunique()
    return distinct_values[distinct_values == len(dataf)].index.tolist()
