from typing import Optional, Any, Callable, Union

import sys
import decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import lightgbm as ltb
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as _LinearRegression


class SPPSF_Polynomial_Time_Model:

    def __init__(self):
        self.Time_Model = None
        self.pred_data = None

    def fit(
        self,
        SPPSF_: np.ndarray,
        Time_:  np.ndarray,
        return_model: Optional[bool] = False
    ) -> Optional[np.ndarray]:

        # Creates df for SPPSF and Months
        modelData = pd.DataFrame(dict(
            SPPSF=SPPSF_,
            const=1,
            Months=Time_
        ))

        bestTimeModel = None
        for degree in range(1, len(modelData['Months'].unique())):
            if degree > 1:
                # modelData['Months' + str(degree)] = (
                #     modelData['Months'] ** degree
                # )
                modelData['Months' + str(degree)] = (
                    modelData['Months'].pow(degree).copy()
                )
                modelData = modelData.copy()
            m = sm.OLS(
                modelData['SPPSF'],
                modelData.drop('SPPSF', axis=1)
            ).fit()

            if bestTimeModel is None or m.bic < bestTimeModel.bic:
                bestTimeModel = m

        self.Time_Model = bestTimeModel

        # this is not being used for anything.
        # modelDataR = pd.DataFrame(
        #     bestTimeModel.model.data.exog,
        #     columns=bestTimeModel.model.data.param_names,
        #     index=bestTimeModel.model.data.row_labels
        # )

        self.pred_data = predData = pd.DataFrame(
            dict(const=1, Months=range(0, modelData['Months'].max() + 1)),
            index=range(0, modelData['Months'].max() + 1)
        )

        if bestTimeModel.df_model > 1:
            for x in range(2, int(bestTimeModel.df_model) + 1):
                predData['Months' + str(x)] = predData['Months'] ** x

        if return_model is True:
            return bestTimeModel

    def Display_Time_Trend(self) -> None:
        plt.plot(
            -self.pred_data['Months'],
            self.Time_Model.predict(self.pred_data),
            '-',
            color='red',
            linewidth=3
        )
        plt.axhline(
            self.Time_Model.predict(self.pred_data).loc[0],
            color='k',
            linestyle='--'
        )
        plt.show()

    def Adjustment_Rate_Return(
        self,
        as_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, str]:

        if self.pred_data is None:
            raise RuntimeError(
                'You need to call "SPPSF_Polynomial_Time_Model.fit" to train '
                'the model before being able to collect data from it'
            )

        if as_pandas is True:
            predDataResults = self.pred_data.copy()
            predDataResults['Model_Prediction'] = (
                self.Time_Model.predict(self.pred_data)
            )

            predDataResults['AdjustMent_Rate'] = (
                self.Time_Model.predict(self.pred_data).loc[0] /
                self.Time_Model.predict(self.pred_data)
            )
            return predDataResults[['Months', 'AdjustMent_Rate']]

        else:
            rateTable = (
                self.Time_Model.predict(self.pred_data).loc[0] /
                self.Time_Model.predict(self.pred_data)
            )
            rateTable.name = 'AdjRate'
            return rateTable

    def trend_summary(self):
        if self.Time_Model is None:
            raise RuntimeError(
                'You need to call "SPPSF_Polynomial_Time_Model.fit" to train '
                'the model before being able to collect data from it'
            )

        return self.Time_Model.summary()

    def to_dict(self, 
        precision: Optional[int] = sys.float_info.dig,
        factor_type: Optional[Callable[[Union[int, float]], Any]] = decimal.Decimal):

        def _decimal_wrapper(num):
            return decimal.Decimal(str(num))

        # this is done because the conversion from an int or a float 
        # to a decimal.Decimal requires one 
        # additional type conversion and that is to a str
        if factor_type == decimal.Decimal:
            factor_type = _decimal_wrapper 
        
        data = self.Adjustment_Rate_Return(as_pandas=True)
        return {
            month: factor_type(round(data['AdjustMent_Rate'][i] ,precision))
            for i, month in enumerate(data['Months'])
        }


class SPPSF_Machine_Learning_Time_Model:

    def __init__(
        self,
        attrs: Optional[dict] = None,
        model_Type: Optional[str] = 'Random Forest',
        Return_Gaussian_Smoothing: Optional[bool] = False,
        Smoothing_Sigma: Optional[int] = 2,
        model_params: Optional[dict] = None
    ):

        if model_params is None:
            
            model_params = {'random_state': 42, 'min_child_samples': 20}

            if ( model_Type == 'LGBM' ):
                model_params['verbose'] = -1

        self.attrs = attrs
        self.model_Type = model_Type
        self.Time_Model = None
        self.pred_data = None

        if (
            model_Type == 'Random Forest' and
            'min_child_samples' in model_params
        ):
            model_params['min_samples_leaf'] = model_params['min_child_samples']
            del model_params['min_child_samples']

        if (
            model_Type == 'LGBM' and
            'min_samples_leaf' in model_params
        ):
            model_params['min_child_samples'] = model_params['min_samples_leaf']
            del model_params['min_samples_leaf']

        self.model_params = model_params
        self.Return_Gaussian_Smoothing = Return_Gaussian_Smoothing
        self.Smoothing_Sigma = Smoothing_Sigma

    def fit(
        self,
        SPPSF_: np.array,
        Time_: np.array,
        return_model: Optional[bool] = False
    ) -> Optional[Union[RandomForestRegressor, ltb.LGBMRegressor]]:

        # Creates df for SPPSF and Months
        Time_ = (
            Time_.rename(columns={Time_.columns.tolist()[0]: "Months"}).copy()
        )

        # not being used
        # Time_ML_Model = None

        if self.model_Type == 'Random Forest':
            rf_tt = RandomForestRegressor(**self.model_params)
            rf_tt.fit(Time_, SPPSF_)
            Time_ML_Model = rf_tt
        else:
            lgbm_tt = ltb.LGBMRegressor(**self.model_params)
            lgbm_tt.fit(Time_, SPPSF_)
            Time_ML_Model = lgbm_tt

        self.Time_Model = Time_ML_Model

        self.pred_data = pd.DataFrame(
            dict(Months=range(0, Time_.max().iloc[0] + 1)),
            index=range(0, Time_.max().iloc[0] + 1)
        )

        if return_model is True:
            return self.Time_Model

    def Display_Time_Trend(self) -> None:
        plt.plot(
            -self.pred_data['Months'],
            self.Time_Model.predict(self.pred_data[['Months']]),
            '-',
            color='red',
            linewidth=3,
            label="Machine Learning Time Trend"
        )

        if self.Return_Gaussian_Smoothing is True:
            plt.plot(
                -self.pred_data['Months'],
                gaussian_filter1d(
                    self.Time_Model.predict(self.pred_data[['Months']]),
                    sigma=self.Smoothing_Sigma
                ),
                color='blue',
                linewidth=3,
                label="Time Trend with Gaussian Smoothing"
            )

        plt.axhline(
            self.Time_Model.predict(self.pred_data)[0],
            color='k',
            linestyle='--',
            label="Current Date Reference"
        )
        plt.legend()
        plt.grid()
        plt.show()

    def Adjustment_Rate_Return(
        self,
        as_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, pd.Series]:

        if self.Time_Model is None:
            raise RuntimeError(
                'You need to call "SPPSF_Polynomial_Time_Model.fit" to train '
                'the model before being able to collect data from it'
            )

        if as_pandas is True:
            predDataResults = self.pred_data.copy()
            predDataResults['Model_Prediction'] = (
                self.Time_Model.predict(self.pred_data[['Months']])
            )
            predDataResults['AdjustMent_Rate'] = (
                self.Time_Model.predict(self.pred_data[['Months']])[0] /
                self.Time_Model.predict(self.pred_data[['Months']])
            )

            if self.Return_Gaussian_Smoothing is True:
                smth = gaussian_filter1d(
                    self.Time_Model.predict(self.pred_data[['Months']]),
                    sigma=self.Smoothing_Sigma
                )

                predDataResults['AdjustMent_Rate_Smoothed'] = smth[0] / smth

                return predDataResults[
                    ['Months', 'AdjustMent_Rate', 'AdjustMent_Rate_Smoothed']
                ]

            else:
                return predDataResults[['Months', 'AdjustMent_Rate']]

        else:
            # not being used
            # rateTable = None
            if self.Return_Gaussian_Smoothing is False:
                rateTable = pd.Series(
                    self.Time_Model.predict(self.pred_data[['Months']])[0] /
                    self.Time_Model.predict(self.pred_data[['Months']])
                )
                rateTable.name = 'AdjRate'
            else:
                smth = gaussian_filter1d(
                    self.Time_Model.predict(self.pred_data[['Months']]),
                    sigma=self.Smoothing_Sigma
                )
                rateTable = pd.Series(smth[0] / smth)
                rateTable.name = 'AdjRateSmoothed'

            return rateTable

    def trend_summary(self) -> dict:
        if self.Time_Model is None:
            raise RuntimeError(
                'You need to call "SPPSF_Polynomial_Time_Model.fit" to train '
                'the model before being able to collect data from it'
            )

        return self.Time_Model.get_params()

    def to_dict(self, 
        precision: Optional[int] = sys.float_info.dig,
        factor_type: Optional[Callable[[Union[int, float]], Any]] = decimal.Decimal ):
        def _decimal_wrapper(num):
            return decimal.Decimal(str(num))

        # this is done because the conversion from an int 
        # or a float to a decimal.Decimal requires one 
        # additional type conversion and that is to a str
        if factor_type == decimal.Decimal:
            factor_type = _decimal_wrapper
        
        data = self.Adjustment_Rate_Return(as_pandas=True)
        if self.Return_Gaussian_Smoothing is False:
            return {
                month: factor_type(round(data['AdjustMent_Rate'][i] ,precision))
                for i, month in enumerate(data['Months'])
            }

        return {
            month: factor_type(round(data['AdjustMent_Rate_Smoothed'][i] ,precision))
            for i, month in enumerate(data['Months'])
        }


def _single_column(values: Any, name: str) -> np.ndarray:
    """
    Convert one column of input values to a 1-D float numpy array.

    Accepts a numpy array, list, pandas Series, or a one-column DataFrame.

    Parameters
    ----------
    values : array-like
        The values to convert.
    name : str
        The argument name, used in the error message.

    Returns
    -------
    numpy.ndarray
        A 1-D float array.

    Raises
    ------
    ValueError
        If ``values`` has more than one column.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a single column of values, but it has shape "
            f"{array.shape}."
        )
    return array


class MLR_Time_Trend:
    """
    Linear regression time trend.

    Fits a straight-line trend of a price-level measure (typically sale price
    per square foot) against the number of months before the target date, and
    turns it into monthly time-adjustment factors.

    The factors have the same format as those of
    ``SPPSF_Polynomial_Time_Model`` and ``SPPSF_Machine_Learning_Time_Model``,
    so the three time models can be used interchangeably.

    Notes
    -----
    Months are counted backwards from the target date: month 0 is the target
    month (usually the most recent month), and month ``t`` is ``t`` months
    earlier. The fitted line is::

        predicted value(t) = intercept + slope * t

    and the adjustment factor for a sale in month ``t`` is::

        factor(t) = predicted value(0) / predicted value(t)

    Multiplying a month-``t`` sale price by its factor adjusts it to month-0
    market levels.

    Despite the class name, the model has a single explanatory variable
    (months). The name is kept so existing code keeps working.

    Examples
    --------
    >>> from pyRealEstate import Time_Trending as tt
    >>> model = tt.MLR_Time_Trend()
    >>> model.fit(sales["SPPSF"], sales["Months"])
    >>> factors = model.Adjustment_Rate_Return(as_pandas=True)
    """

    def __init__(self):
        self.x_range = None
        self.pred_data = None
        self.Time_Model = None

    def _check_fitted(self) -> None:
        """Raise a clear error if ``fit`` has not been called yet."""
        if self.Time_Model is None:
            raise RuntimeError(
                'You need to call "MLR_Time_Trend.fit" to train '
                'the model before being able to collect data from it'
            )

    def fit(
        self,
        SPPSF_: np.ndarray,
        Time_: np.ndarray,
        return_model: bool = False
    ) -> Optional[_LinearRegression]:
        """
        Fit the linear time trend.

        Parameters
        ----------
        SPPSF_ : array-like
            A price-level measure for each sale, typically sale price per
            square foot (SPPSF). A ratio such as sale price / prior assessed
            value also works.
        Time_ : array-like
            For each sale, the number of months between the sale and the
            target month (month 0). A numpy array, list, pandas Series, or
            one-column DataFrame. Whole or fractional months are accepted.
        return_model : bool, default False
            If True, return the fitted scikit-learn model.

        Returns
        -------
        sklearn.linear_model.LinearRegression or None
            The fitted model when ``return_model`` is True, otherwise None.
            Its slope and intercept are ``coef_[0]`` and ``intercept_``.

        Raises
        ------
        ValueError
            If ``SPPSF_`` or ``Time_`` has more than one column.
        """
        sppsf = _single_column(SPPSF_, "SPPSF_")
        months = _single_column(Time_, "Time_")

        # scikit-learn expects the explanatory variable as a 2-D column.
        self.Time_Model = model = _LinearRegression()
        model.fit(months.reshape(-1, 1), sppsf)

        # Predict for every whole month from the target month (0) to the
        # oldest month in the data. int64 on every platform, like the other
        # time models.
        self.x_range = np.arange(0, int(np.ceil(months.max())) + 1, dtype=np.int64)
        self.pred_data = model.predict(self.x_range.reshape(-1, 1))

        if return_model is True:
            return self.Time_Model

    def Adjustment_Rate_Return(
        self,
        as_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Monthly time-adjustment factors.

        Parameters
        ----------
        as_pandas : bool, default False
            If True, return a DataFrame with the columns ``Months`` and
            ``AdjustMent_Rate``. Otherwise return a Series named ``AdjRate``,
            indexed by month.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The factor ``predicted value(0) / predicted value(t)`` for every
            whole month ``t`` from 0 to the oldest month in the data.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_fitted()

        adjustment_rate = self.pred_data[0] / self.pred_data

        if as_pandas is True:
            return pd.DataFrame({
                'Months': self.x_range,
                'AdjustMent_Rate': adjustment_rate
            })

        rateTable = pd.Series(adjustment_rate)
        rateTable.name = 'AdjRate'
        return rateTable

    def trend_summary(self) -> dict:
        """
        Settings of the fitted scikit-learn model.

        Returns
        -------
        dict
            The model settings from ``LinearRegression.get_params()``. The
            fitted slope and intercept are ``Time_Model.coef_[0]`` and
            ``Time_Model.intercept_``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_fitted()
        return self.Time_Model.get_params()

    def to_dict(self,
        precision: Optional[int] = sys.float_info.dig,
        factor_type: Optional[Callable[[Union[int, float]], Any]] = decimal.Decimal):
        """
        Adjustment factors as a dictionary of ``{month: factor}``.

        Parameters
        ----------
        precision : int, default sys.float_info.dig (15)
            Number of decimal places each factor is rounded to.
        factor_type : callable, default decimal.Decimal
            Type each factor is converted to, for example ``float`` or
            ``decimal.Decimal``.

        Returns
        -------
        dict
            ``{month: factor}`` for every whole month from 0 to the oldest
            month in the data.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        def _decimal_wrapper(num):
            return decimal.Decimal(str(num))
        
        # this is done because the conversion from an int or a float 
        # to a decimal.Decimal requires one 
        # additional type conversion and that is to a str
        if factor_type == decimal.Decimal:
            factor_type = _decimal_wrapper
    
        data = self.Adjustment_Rate_Return(as_pandas=True)
        return {
            month: factor_type(round(data['AdjustMent_Rate'][i] ,precision))
            for i, month in enumerate(data['Months'])
        }
