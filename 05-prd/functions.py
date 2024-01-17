#------- Bibliotecas:

# Manipulação de dados:

import pandas as pd
import numpy as np

# Modelagem:

from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import Naive, AutoARIMA, HoltWinters, AutoETS
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

# Reconciliação

from hierarchicalforecast.methods import BottomUp, OptimalCombination
from hierarchicalforecast.core import HierarchicalReconciliation


#------- Funções:

def read_data(path: str):

    df = pd.read_csv(path)

    return df


def clean_data(df: pd.DataFrame):

    #---- 1. Excluindo a variável de country:

    df = df\
        .drop(columns = 'country')

    #---- 2. Mudando o tipo da variável de date para datetime:

    df['date'] = pd.to_datetime(df['date'])

    #---- 3. Renomeando as variáveis de quantidade de vendas e data:
    # date -> ds
    # quantity -> y

    df = df\
        .rename(columns = {'date': 'ds', 
                           'quantity': 'y'})

    return df

def format_hierarchical_df(df: pd.DataFrame, cols_hierarchical: list):

    #---- 1. Cria uma lista de listas: [[col1], [col1, col2], ..., [col1, col2, coln]]

    hier_list = [cols_hierarchical[:i] for i in range(1, len(cols_hierarchical) + 1)]

    #---- 2. Aplica a função aggregate que formata os dados em que a lib hierarchical pede

    Y_df, S_df, tags = aggregate(df = df, spec = hier_list)

    return Y_df, S_df, tags


def apply_time_series_models(Y_df: pd.DataFrame, 
                             S_df: pd.DataFrame,
                             tags: dict,
                             freq: str,
                             ts_models: None,
                             reconcilers_ts: None,
                             horizon_forecast: int = 30):

    model_ts = StatsForecast(ts_models, 
                             freq = freq,
                             n_jobs = -1)
    model_ts.fit(Y_df)
    
    Y_hat_df_ts = model_ts.forecast(h = horizon_forecast)

    hrec_ts = HierarchicalReconciliation(reconcilers = reconcilers_ts)
    
    Y_rec_df_ts = hrec_ts.reconcile(Y_hat_df = Y_hat_df_ts, 
                                    S = S_df,
                                    tags = tags)

    return Y_rec_df_ts.reset_index()

def apply_machine_learning_models(Y_df: pd.DataFrame, 
                                  S_df: pd.DataFrame,
                                  tags: dict,
                                  freq: str,
                                  ml_models: None, 
                                  lags_ml: list,
                                  date_features_ml: list,
                                  lag_transforms_ml: dict,
                                  reconcilers_ml: None,
                                  horizon_forecast: int = 30):

    model_ml = MLForecast(models = ml_models,
                              freq = freq, 
                              num_threads = 6,
                              lags = lags_ml, 
                              date_features = date_features_ml, 
                              lag_transforms = lag_transforms_ml
                             )

    model_ml.fit(Y_df.reset_index(), id_col = 'unique_id', time_col = 'ds', target_col = 'y')
    
    Y_hat_df_ml = model_ml.predict(h = horizon_forecast)

    hrec_ml = HierarchicalReconciliation(reconcilers = reconcilers_ml)

    Y_rec_df_ml = hrec_ml.reconcile(Y_hat_df = Y_hat_df_ml, 
                            S = S_df,
                            tags = tags)

    Y_rec_df_ml = Y_rec_df_ml[[col for col in Y_rec_df_ml.columns if 'index' not in col]]

    return Y_rec_df_ml.reset_index()


def apply_models(Y_df: pd.DataFrame, 
                 S_df: pd.DataFrame,
                 tags: dict,
                 freq: str,
                 ts_models: None,
                 reconcilers_ts: None,
                 ml_models: None, 
                 lags_ml: None,
                 date_features_ml: None,
                 lag_transforms_ml: None,
                 reconcilers_ml: None,
                 horizon_forecast: None):

    if ts_models:

        print('Executando os modelos de séries temporais...')

        ts_recommendations = apply_time_series_models(Y_df = Y_df,
                                                      S_df = S_df,
                                                      tags = tags,
                                                      freq = freq,
                                                      ts_models = ts_models,
                                                      reconcilers_ts = reconcilers_ts,
                                                      horizon_forecast = horizon_forecast)
    else:

        ts_recommendations = pd.DataFrame(columns = ['ds', 'unique_id'])

    if ml_models:

        print('Executando os modelos de Machine Learning')

        ml_recommendations = apply_machine_learning_models(Y_df = Y_df,
                                                           S_df = S_df,
                                                           tags = tags,
                                                           freq = freq,
                                                           ml_models = ml_models,
                                                           lags_ml = lags_ml,
                                                           date_features_ml = date_features_ml,
                                                           lag_transforms_ml = lag_transforms_ml,
                                                           reconcilers_ml = reconcilers_ml,
                                                           horizon_forecast = horizon_forecast)
    else:

        ml_recommendations = pd.DataFrame(columns = ['ds', 'unique_id'])

    result_df = ts_recommendations.merge(ml_recommendations, on = ['ds', 'unique_id'], how = 'outer')

    return result_df


def clean_recommendations(df_rec: pd.DataFrame):

    model_col = [col for col in df_rec.columns if '/' in col]

    df_rec1 = df_rec[['unique_id', 'ds'] + model_col]\
        .assign(\
            nivel_hierarquia = lambda x: np.where(x['unique_id'].str.count('/') == 0, 1, x['unique_id'].str.count('/') + 1)
        )\
        .query(f'nivel_hierarquia == {len(cols_hierarchical)}')

    df_rec1[cols_hierarchical] = df_rec1['unique_id'].str.split('/', n = len(cols_hierarchical), expand = True)

    df_rec1 = df_rec1\
        .rename(columns = {'ds': 'date'})\
        .drop(columns = ['unique_id', 'nivel_hierarquia'])\
        .reset_index(drop = True)[cols_hierarchical + ['date'] + model_col]
    
    return df_rec1