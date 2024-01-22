from functions import *

#---- 1. Lendo os dados:

path_data = 'https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-hierarchical-forecasting/main/retail-usa-clothing.csv'

dados = read_data(path = path_data)

#---- 2. Criando a função de predict:

def fun_predict(days_to_forecast):
    
  #---- a. Corrigindo os dados:
  print('Corrigindo os dados')

  df = clean_data(df = dados)

  #---- b. Formatando os dados para os modelos:
  print('Formatando os dados para os modelos:')

  cols_hierarchical = ['region', 'state', 'item']

  Y_df, S_df, tags = format_hierarchical_df(df = df, cols_hierarchical = cols_hierarchical)

  #---- c. Aplicando os modelos de TS e ML:
  print('Aplicando os modelos de TS e ML')

  # Modelos: 

  hw = HoltWinters(season_length = 7, error_type = 'M') # Holtwinters com sazonalidade de 7 dias e erro do tipo Aditivo
  lin_reg = LinearRegression() # Regressão linear

  # Features de data:

  @njit
  def rolling_mean_7(x):
      return rolling_mean(x, window_size = 7)

  @njit
  def rolling_mean_14(x):
      return rolling_mean(x, window_size = 14)

  @njit
  def rolling_mean_21(x):
      return rolling_mean(x, window_size = 21)

  @njit
  def rolling_mean_28(x):
      return rolling_mean(x, window_size = 28)

  df_recommendations =  apply_models(Y_df = Y_df,
                                    S_df = S_df,
                                    tags = tags,
                                    freq = 'D',
                                    ts_models = [hw],
                                    reconcilers_ts = [BottomUp()],
                                    ml_models = [lin_reg],
                                    lags_ml = [1, 7, 14, 21, 28, 30],
                                    date_features_ml = ['dayofweek', 'month', 'year', 'quarter', 'day', 'week'],
                                    lag_transforms_ml = {
                                        1: [expanding_mean],
                                        7: [rolling_mean_7],
                                        14: [rolling_mean_14],
                                        21: [rolling_mean_21],
                                        28: [rolling_mean_28],
                                    },
                                    reconcilers_ml = [OptimalCombination(method = 'ols', nonnegative = True)],
                                    horizon_forecast = days_to_forecast)

  print('Corrigindo o dataframe')
  df_result = clean_recommendations(df_rec = df_recommendations, cols_hierarchical = cols_hierarchical)

  return df_result


inputs = gr.Number(label = 'Dias para a projeção', value = 30)
outputs = [gr.DataFrame(headers = dados.columns.tolist())]

demo = gr.Interface(fn = fun_predict, 
                    inputs = inputs,
                    # examples = [dados.head(3)],
                    outputs = outputs,
                    title = 'Projeções de múltiplas séries temporais')

demo.launch(share = True)