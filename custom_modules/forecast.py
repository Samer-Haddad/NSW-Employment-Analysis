import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from typing import Tuple, List
from .style import load_theme
from .plots import check_dir
from .validate import err, validate

warnings.filterwarnings("ignore")
DIR_MODEL = Path('Images/ARIMA Model/')
DIR_FORECAST = Path('Images/Forecast/')
########################################################################

@validate()
def evaluate(dataframe:pd.Series, arima_order:Tuple[int], train_size:float=0.8, log:bool=True, title:str=None, figsize:Tuple[int]=(16, 9), show:bool=False, save_fig:bool=False, dpi:int=300, transparent:bool=False):
   '''
   Train and evaluate an ARIMA model
   setting log=True will apply a log scale on the training data.
   This is recommended to prevent negative predictions as we scale back by an exponent function. 
   '''
   if train_size < 0.0 or train_size > 1.0:
      err('ValueError', 'train_size', train_size, '[0.0 < train_size < 1.0]')
   if len(figsize) != 2:
      err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
   if len(arima_order) != 3:
      err('ValueError', 'arima_order', arima_order, 'Tuple(int, int, int)')

   THEME, _ = load_theme()
   data = np.array(dataframe)
   train_size = int(len(data) * train_size)
   train = data[0:train_size]
   if log: # apply log function on training data
      train = np.log(train)
   test = data[train_size:]
   history = [x for x in train]
   predictions = list()
 
   for t in range(len(test)):
      model = ARIMA(history, order=arima_order)
      model_fit = model.fit()
      yhat = model_fit.forecast()[0]
      predictions.append(yhat)
      if log:
         history.append(np.log(test[t]))
      else:
         history.append(test[t])

   pred_mse = [x for x in predictions]
   
   if log: # scale back
      predictions = np.exp(predictions)

   plt.ioff()
   fig, ax = plt.subplots(figsize=figsize, layout='constrained')
   x_axis = dataframe.index.to_list()
   ax.set(xticks=x_axis, xticklabels=x_axis)
   pred_axis = x_axis[train_size-1:]
   pred_line = [data[train_size-1]]
   for x in predictions:
      pred_line.append(x)

   ax.plot(x_axis, data, color=THEME[3], label='History', marker='o')
   ax.plot(pred_axis, pred_line, color=THEME[0], label='Predicted', linestyle='dashed', marker='o')

   plt.ylabel('Headcount', labelpad=25, fontsize=14)
   plt.xlabel('Year', labelpad=25, fontsize=14)
   plt.title(f'ARIMA Model Fit\n[{title}] {arima_order}', pad=50, fontsize=18)
   plt.legend()
   if show: plt.show()
   if save_fig:
        check_dir(DIR_MODEL)
        save_title = f'ARIMA Model Fit [{title}] {arima_order}.png'
        path = Path.joinpath(DIR_MODEL, save_title)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
   plt.close(fig)
   
   if log:
      test = np.log(test)

   error = mean_squared_error(test, pred_mse)
   return error
########################################################################

@validate()
def grid_search(dataframe:pd.Series, p_values:range, d_values:range, q_values:range, train_size:float=0.8, log:bool=True, title:str=None, show:bool=False, save_fig:bool=False):
   '''
   Grid Search in the given space of (p_values, d_values, and q_values) for optimal or near optimal parameters.
   '''
   if train_size < 0.0 or train_size > 1.0:
      err('ValueError', 'train_size', train_size, '[0.0 < train_size < 1.0]')

   df = dataframe.astype('float')
   best_score, best_cfg = float("inf"), None
   for p in p_values:
      for d in d_values:
         for q in q_values:
            order = (p,d,q)
            try:
               mse = evaluate(df, order, train_size=train_size, log=log)
               if mse < best_score:
                  best_score, best_cfg = mse, order
            except:
               continue
   title += f'] [train={train_size}'
   mse = evaluate(df, best_cfg, train_size=train_size, log=log, title=title, show=show, save_fig=save_fig)
   return best_cfg
########################################################################

@validate()
def projection(dataframe:pd.Series, arima_order:Tuple[int], years:int, log:bool=True, title:str=None, figsize:Tuple[int]=(16, 9), show:bool=False, save_fig:bool=False, dpi:int=300, transparent:bool=False):
   '''
   Make predictions for a given number of years
   '''
   if len(figsize) != 2:
      err('ValueError', 'figsize', figsize, 'Tuple(int, int)')
   if len(arima_order) != 3:
      err('ValueError', 'arima_order', arima_order, 'Tuple(int, int, int)')

   THEME, _ = load_theme()

   df = dataframe.astype('float')
   x_axis = df.index.to_list()
   df = np.array(df)
   history = [x for x in df]
   if log:
      history = np.log(history)
   predictions = list()

   model = ARIMA(history, order=arima_order)
   model_fit = model.fit()
   predictions = model_fit.predict(start=len(history), end=len(history)+years-1, typ='levels')

   if log: # scale back
      predictions = np.exp(predictions)

   last_year = x_axis[-1]
   model_axis = list(range(last_year, last_year+years+1))

   plt.ioff()
   fig, ax = plt.subplots(figsize=figsize, layout='constrained')
   x_range = list(range(x_axis[0], x_axis[-1]+years+1))
   ax.set(xticks=x_range, xticklabels=x_range)
   plt.title(f'Trend Projection\n[{title}]', pad=50, fontsize=18)

   pred_line = [df[-1]]
   for x in predictions:
      pred_line.append(x)

   ax.plot(x_axis, df, color=THEME[3], label='History', marker='o')
   ax.plot(model_axis, pred_line, color=THEME[0], label='Forecast', linestyle='dashed', marker='o')

   plt.ylabel('Headcount', labelpad=25, fontsize=14)
   plt.xlabel('Year', labelpad=25, fontsize=14)
   plt.legend()
   if show: plt.show()
   if save_fig:
      check_dir(DIR_FORECAST)
      save_title = f'Trend Projection [{title}].png'
      path = Path.joinpath(DIR_FORECAST, save_title)
      fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent)
   plt.close(fig)

   return [int(np.round(x)) for x in predictions]
########################################################################