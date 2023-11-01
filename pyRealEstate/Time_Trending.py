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
            dict(Months=range(0, Time_.max()[0] + 1)),
            index=range(0, Time_.max()[0] + 1)
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


class MLR_Time_Trend:
    """
    Multiple Linear Regression Time Trend

    This class will produce factors that are to be used on the
    sale price of a property to adjust that property to a point in time that
    is the month after the lowest month that was used to train the model.
    """

    def __init__(self):
        self.x_range = None
        self.pred_data = None
        self.Time_Model = None

    def fit(
        self,
        SPPSF_: np.ndarray,
        Time_: np.ndarray,
        return_model: bool = False
    ) -> Optional[_LinearRegression]:
        """
        Trains the time model

        :param SPPSF_: an array of (sale_price / last_total_value) ratios
        :type SPPSF_: :py:class:`np.ndarray`

        :param Time_: an array of the number of months from the
                      last_total_value was done and the sale month

        :type Time_: :py:class:`np.ndarray`

        :param return_model: return the model object
        :type return_model: bool

        :return: either `None` of the model object
        :rtype: Optional - `None` or
                :py:class:`sklearn.linear_model.LinearRegression`
        """

        modelData = pd.DataFrame(dict(
            SPPSF=SPPSF_.copy(),
            Months=Time_.copy()
        ))

        if modelData['Months'].ndim < 2:
            modelData['Months'] = modelData['Months'].reshape(-1, 1)

        self.Time_Model = model = _LinearRegression()
        model.fit(modelData['Months'], modelData['SPPSF'])

        self.x_range = x_range = np.linspace(
            modelData['Months'].min(),
            modelData['Months'].max()
        )
        self.pred_data = model.predict(x_range.reshape(-1, 1))

        if return_model is True:
            return self.Time_Model

    def Adjustment_Rate_Return(
        self,
        as_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Gets the factor adjustments

        This function can only be called after the model has been trained.
        To train the model use the :py:func:`MLR_Time_Trend.fit` method

        :param as_pandas: To return a :py:class:`pd.DataFrame` object.
                          This object will have 2 columns in it one labeled
                          "Months" and the other "AdjustMent_Rate"
        :type as_pandas: bool

        :return: adjustment factors
        :rtype:  :py:class:`pd.DataFrame` or :py:class:`pd.Series`
        """
        if self.Time_Model is None:
            raise RuntimeError(
                'You need to call "SPPSF_Polynomial_Time_Model.fit" to train '
                'the model before being able to collect data from it'
            )

        if as_pandas is True:
            df = pd.DataFrame()
            df['Months'] = self.x_range.copy()
            df['AdjustMent_Rate'] = self.pred_data.copy()

            return df[['Months', 'AdjustMent_Rate']]

        rateTable = pd.Series(self.pred_data.copy())
        rateTable.name = 'AdjRate'
        return rateTable

    def trend_summary(self) -> dict:
        """
        Trend Summary

        This function can only be called after the model has been trained.
        To train the model use the :py:func:`MLR_Time_Trend.fit` method

        :return: parameters used in the model.
        :rtype: dict
        """
        return self.Time_Model.get_params()
    
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
