import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt


class SPPSF_Polyniomial_Time_Model :


  def fit( self ,SPPSF_ , Time_ , return_model =False ):
    #Creates df for SPPSF and Months
    modelData = pd.DataFrame({
                              'SPPSF' : SPPSF_ , 
                              'const' : 1, 
                              'Months' : Time_})

    bestTimeModel = None
    for degree in range(1, len(modelData['Months'].unique())):
      if degree > 1:
          #modelData['Months' + str(degree)] = modelData['Months'] ** degree
          modelData['Months' + str(degree)] = modelData['Months'].pow(degree).copy()
          modelData = modelData.copy()
      m = sm.OLS(modelData['SPPSF'], modelData.drop('SPPSF', axis = 1)).fit()

      if(bestTimeModel == None or m.bic < bestTimeModel.bic):
          bestTimeModel = m

    self.Time_Model = bestTimeModel

    modelDataR = pd.DataFrame(bestTimeModel.model.data.exog
                              , columns = bestTimeModel.model.data.param_names
                              , index = bestTimeModel.model.data.row_labels)

    predData = pd.DataFrame({
                               'const' : 1
                               , 'Months' : range(0, modelData['Months'].max() + 1)} , index = range(0, modelData['Months'].max() + 1))

    if bestTimeModel.df_model > 1:
      for x in range(2, int(bestTimeModel.df_model) + 1):
        predData['Months' + str(x)] = predData['Months'] ** x
    self.pred_data = predData

    if return_model ==True:
      return bestTimeModel
  
  def Display_Time_Trend(self): 
    plt.plot(-self.pred_data['Months'], self.Time_Model.predict(self.pred_data), '-', color = 'red', linewidth = 3)
    plt.axhline(self.Time_Model.predict(self.pred_data).loc[0], color='k', linestyle='--')
    plt.show()

  def Adjustment_Rate_Return(self, as_pandas =False):
    if as_pandas == True :
      predDataResults = self.pred_data.copy()
      predDataResults['Model_Prediction'] = self.Time_Model.predict(self.pred_data)
      predDataResults['AdjustMent_Rate'] = self.Time_Model.predict(self.pred_data).loc[0] / self.Time_Model.predict(self.pred_data)
      return predDataResults[['Months','AdjustMent_Rate']]

    else:
      rateTable = self.Time_Model.predict(self.pred_data).loc[0] / self.Time_Model.predict(self.pred_data)
      rateTable.name = 'AdjRate'
      return rateTable
    
  def trend_summary(self):
    return self.Time_Model.summary()
