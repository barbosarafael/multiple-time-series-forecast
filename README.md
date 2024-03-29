# Projeção de múltiplas séries temporais

Nesse repositório vou mostrar como podemos fazer projeções múltiplas séries temporais de uma única vez. 

# Índice

- [1. Dados utilizados](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#1-dados-utilizados)
- [2. Projeções hierárquicas (hierarchical forecast)](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#2-proje%C3%A7%C3%B5es-hier%C3%A1rquicas-hierarchical-forecast)
  - [2.1. Situação hipotética](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#21-situa%C3%A7%C3%A3o-hipot%C3%A9tica)
  - [2.2. Um pouquinho de teoria (na prática)](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#22-um-pouquinho-de-teoria-na-pr%C3%A1tica)
  - [2.3. Reconciliação](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#23-reconcilia%C3%A7%C3%A3o)
- [3. Explorações iniciais](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#3-explora%C3%A7%C3%B5es-iniciais)
- [4. Processo de modelagem](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#4-processo-de-modelagem)
  - [4.1. Organização dos dados](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#41-organiza%C3%A7%C3%A3o-dos-dados)
  - [4.2. Separação em treino e validação](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#42-separa%C3%A7%C3%A3o-em-treino-e-valida%C3%A7%C3%A3o)
  - [4.3. Aplicação dos modelos](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#43-aplica%C3%A7%C3%A3o-dos-modelos)
  - [4.4. Predict do modelo](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#44-predict-do-modelo)
  - [4.5. Reconciliação](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#45-reconcilia%C3%A7%C3%A3o)
- [5. Avaliação dos modelos](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#5-avalia%C3%A7%C3%A3o-dos-modelos)
  - [5.1. RMSE](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#51-rmse)
  - [5.2. Comparação gráfica](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#52-compara%C3%A7%C3%A3o-gr%C3%A1fica)
- [6. Tabela final](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#6-tabela-final)
- [7. Bônus: Feature importance para os modelos de Machine Learning](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#7-b%C3%B4nus-feature-importance-para-os-modelos-de-machine-learning)
- [8. Modelo em produção](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#8-modelo-em-produ%C3%A7%C3%A3o)
- [9. Referências](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#9-refer%C3%AAncias)
- [10. Possíveis melhorias (para outras pessoas):](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main?tab=readme-ov-file#10-poss%C3%ADveis-melhorias-para-outras-pessoas)


## 1. Dados utilizados

Os dados se referem a venda de roupa no varejo dos USA. Os dados raw podem ser encontrados diretamente nesse [link](https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-hierarchical-forecasting/main/retail-usa-clothing.csv). Abaixo um print da tabela.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/ef98a6d7-5d37-4037-b0f3-323f07b4bae1)

**Nosso objetivo é projetar as vendas de roupas no varejo de todas as "combinações/concatenação" das 3 variáveis do dataset: state, item e region**. 

*Obs.: A variável **country** sempre será USA, por isso não iremos considerá-la no estudo.*

## 2. Projeções hierárquicas (hierarchical forecast)

### 2.1. Situação hipotética

Pensa que tu trabalhas nessa empresa de varejo e o seu chefe pediu para você fazer a projeção de quantas vendas terão para os itens que vocês vendem (roupas feminas e masculinos, sapatos, etc...). Você deve imaginar, ok, são apenas 5 ou 6 itens para projetar, dá para modelar cada série na mão, individualmente. 

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/cdc477c3-1814-4339-86c7-fabc050d8b60)

**Mas** aí você encontra sua primeira dificuldade: as lojas estão espalhadas pelos estados do USA. Logo, agora você não tem somente os 6 itens para modelar e sim os 6 itens dentro de cada estado no banco.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/c963677b-f4b0-4bbf-9bae-948df13d2d9e)

Agora teremos que projetar 108 (6 itens $\times$ 18 estados) séries temporais diferentes. Já pensa ter que modelar uma a uma? Na mão?

> Adendo e provocação: Você pode aplicar métodos automáticos de projeção para cada série, como um AutoARIMA da vida. Certíssimo, eu faria isso. Mas e se o seu chefe te pedir para agregar as projeções somente do estado de Nova York? Será que, quando você fazer a agregação para o nível de estado, as projeções irão ser exatamente iguais?

**Entrando** aí a sua segunda (e última) dificuldade. Não basta ter que projetar as vendas dos itens dentro de cada Estado. Cada estado, vai estar dentro de uma região... Logo, você tem mais um nível de projeção, mas esse não é tão dificultoso assim...

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/7dd5ee75-0666-44f4-8c64-6ffa1650a834)

O lado positivo dessa "última dificuldade" é que um Estado só pode estar dentro de uma Região. Logo, não teremos novas combinações. 

**Para** o nosso caso, temos mais um lado positivo. O gráfico abaixo mostra quantas combinações distintas entre as 3 variáveis foram vendidas durante o tempo, ou seja, quantas projeções de vendas teremos que fazer diariamente. Sempre foram 91 combinações e não aquelas 108 que havia comentado anteriormente. Em termos práticos, em alguns estados não são vendidos determinados itens.

![teste](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/cb813600-7aef-4a8a-b0e4-4eefa698a039)

### 2.2. Um pouquinho de teoria (na prática)

De forma visual e resumida, a nossa hierarquia de variáveis pode ser representada da seguinte maneira:

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/8211ebe8-27c0-41e2-b33a-8a70da774157)

Agora vamos supor que **rodamos modelos** para projetar as vendas para as séries, dentro de cada nível, isto é, vamos fazer as projeções para cada um dos valores que temos dentro dos níveis. Exemplos abaixo:

- Nível I: SouthCentral
- Nível II: SouthCentral/Vermont, SouthCentral/Maine e SouthCentral/Connecticut
- Nivel III:
  - SouthCentral/Vermont/Roupas Femininas, SouthCentral/Vermont/Roupas Masculinas e SouthCentral/Vermont/Roupas Infantil
  - SouthCentral/Maine/Roupas Femininas, SouthCentral/Maine/Roupas Masculinas e SouthCentral/Maine/Roupas Infantil
  - SouthCentral/Connecticut/Roupas Femininas, SouthCentral/Connecticut/Roupas Masculinas e SouthCentral/Connecticut/Roupas Infantil
 
Infelizmente, os valores não batem na hora de agregar os valores nos diferentes níveis. Para o nosso caso, estamos falando que a projeção das vendas de todos os **Estados na Região SouthCentral** não batem com as vendas da **Região SouthCentral**.

```python
Y_hat_df\
    .reset_index()\
    .assign(\
        nivel_hierarquia = lambda x: np.where(x['unique_id'].str.count('/') == 0, 1, x['unique_id'].str.count('/') + 1)
    )\
    .groupby('nivel_hierarquia')[Y_hat_df.select_dtypes(include = 'number').columns]\
    .sum()
```

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/8d103b66-24ed-4b59-b0dc-8211b44e6897)

Na imagem, notem que as resultados das projeções dos níveis não batem. Não achei um motivo específico para isso, minha hipótese é que isso acontença devido a independência das séries devido aos níveis. 

**Para** essas projeções de diferentes níveis baterem, utilizamos a **Reconciliação**:

### 2.3. Reconciliação

A Reconciliação é o método que vai fazer a soma das projeções dos diferentes níveis baterem. Ele é um processo após a projeção, como podemos ver na imagem abaixo.

<p>
    <img src="https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/1a2c4a23-e417-44cf-ad33-c4efc5c35a26" alt>
    <em>Retirado de: Hierarchical TimeSeries Reconciliation by Adrien</em>
</p>

Existem alguns métodos de reconciliação. Os principais são:

#### BottomUp

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/ebfd2997-59ba-4d37-84fe-ff162016be18)

A intuição por trás desse método é, que após a projeção, começaremos a "reestimar" as projeções a partir dos menores níveis até chegar no maior nível. 

#### TopDown

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/c8f2d96c-1ecf-427e-8007-cce8b31e1fb4)

Ao contrário do BottomUp, o TopDown começa o processo de "reestimação" das projeções a partir do maior nível até chegar no menor nível. 

#### MiddleOut

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/aaceeeb5-2224-492b-b01e-1258ffb6e060)

As "reestimações" desse método começam a partir de algum nível intermediário. Para o nosso caso, esse nível deveria ser o nível de Estado (Região/Estado, no código). 

#### Outras

Existem outros métodos que mudam a estratégia de como esse processo de "reestimação" vai acontecer e outras focadas em diminuir o erro. Algumas delas são:

- OptimalCombination
- MinTrace
- ERM
- PERMBU
- Normality
- Bootstrap

## 3. Explorações iniciais

Apenas alguns highlights:

- As datas mais recentes são as que possuem mais vendas
- Tennessee (1.351.608), Kentucky (1.298.425) e California (1.173.635) são os estados com mais vendas
- Vestimentas de mulheres são os itens mais vendidos (11.283.595)
- SouthCentral é a região/regional com maior quantidade de vendas (4.792.847)
- Tennessee/womens_clothing/SouthCentral foi a combinação (variáveis region/state/item) que teve mais vendas durante o período 
- Tendência no aumento das vendas no varejo, perceptível a partir dos anos 2000 (ver imagem abaixo)

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/5bb81ca3-3402-4207-825e-9f7a137b3da5)

Para análise mais detalhadas, ver os notebooks da pasta [01-explorations](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main/02-notebooks/01-explorations).

## 4. Processo de modelagem

### 4.1. Organização dos dados

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/ef98a6d7-5d37-4037-b0f3-323f07b4bae1)

As bibliotecas `StatsForecast` e `MLForecast` exigem que os dados estejam formatados de uma maneira bastante específica. 

A primeira modificação é renomear as colunas de data e quantidade de vendas:

```python
#---- 3. Renomeando as variáveis de quantidade de vendas e data:
# date -> ds
# quantity -> y

df = df\
    .rename(columns = {'date': 'ds', 
                       'quantity': 'y'})
```

A segunda modificação é o formato dos dados. Para isso, vamos utilizar a função `aggregate`. Ela exige dois parâmetros: um dataframe com os dados (ver print acima) e uma lista de lista das hierarquias `[['region'], ['region', 'state'], ['region', 'state', 'item']]`. 

Criei uma função para deixar isso de forma mais simples. 


```python
def format_hierarchical_df(df: pd.DataFrame, cols_hierarchical: list):

    #---- 1. Cria uma lista de listas: [[col1], [col1, col2], ..., [col1, col2, coln]]

    hier_list = [cols_hierarchical[:i] for i in range(1, len(cols_hierarchical) + 1)]

    #---- 2. Aplica a função aggregate que formata os dados em que a lib hierarchical pede

    Y_df, S_df, tags = aggregate(df = df, spec = hier_list)

    return Y_df, S_df, tags
```

Ela retorna 3 objetos. 

**Y_df**: dataframe hierarquico. Notem que foi criado uma série para cada possibilidade dentro dos níveis que havia comentado.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/968fd956-a13d-48d7-9d5d-0bb9ca6dd45a)

**tags**: dicionário onde as chaves são os níveis (region, region/state e region/state/item) e as chaves as combinações dentro desses níveis.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/02a58d6e-fd23-4edc-984f-c577a880d604)

**S_df**: dataframe/matriz que ajuda na hora de fazer a reconciliação. Identifica se um nível pertence ao outro.

### 4.2. Separação em treino e validação

Sem muito rodeio aqui:

- **Treino**: 25/11/1997 a 31/12/2008
- **Validação**: 01/01/2009 a 28/07/2009 (quase 8 meses)

Teremos um **Y_train_df** e um **Y_valid_df**.

### 4.3. Aplicação dos modelos

Ao total, vamos testar 8 modelos:

- **Baseline**: Naive
- **Clássicos de séries temporais**
  1. HoltWinters
  2. ARIMA
  3. ETS
- **Machine Learning**
  1. Regressão Linear
  2. Árvore de decisão: com e sem tuning
  3. Random Forest: com e sem tuning
  4. LightGBM: com e sem tuning
  5. XGBoost: com e sem tuning

#### Séries Temporais

Iremos utilizar a lib `statsforecast`, que contém os modelos. O notebook com esses resultados podem ser encontrados em [01-baseline_and_ts_models.ipynb](https://github.com/barbosarafael/multiple-time-series-forecast/blob/main/02-notebooks/02-baseline_and_ts/01-baseline_and_ts_models.ipynb). Primeiro, vamos definir os modelos que iremos utilizar e instanciá-los.

```python
from statsforecast import StatsForecast
from statsforecast.models import Naive, AutoARIMA, HoltWinters, AutoETS

naive = Naive() # baseline
arima = AutoARIMA(season_length = 7) # ARIMA com sazonalidade de 7 dias
hw = HoltWinters(season_length = 7, error_type = 'M') # Holtwinters com sazonalidade de 7 dias e erro do tipo Aditivo
ets = AutoETS(season_length = 7) # ETS com sazonalidade de 7 dias

model = StatsForecast(models = [naive, arima, hw, ets], freq = 'D', n_jobs = -1)
model.fit(Y_train_df)
```

Notem que passei um parâmetro de identificação de sazonalidade de 7 dias. 

Após isso, passamos para a função `StatsForecast` os modelos, a frequência, que será Diária, e quantos cores de CPU serão utilizados para o processo, -1 significa todos. E no fim o `.fit` aplicando os modelos nos dados de treino.

No fim ele mostra somente o seguinte output:

```
StatsForecast(models=[Naive,AutoARIMA,HoltWinters,AutoETS])
```

> Na documentação da lib, eles aceitam dataframes do Pandas, Spark, Dask e Ray. Li em algum local que estão começando a desenvolver para aceitar em Polars também.

#### Machine Learning

Para a parte de ML, teremos um mix de bibliotecas dos modelos (qualquer um que seja `.fit` e `.transform`), como o `scikit-learn`, `lightgbm` e `xgboost`. E também temos a `MLForecast` que irá fazer o processo de modelagem. 

Mas antes de aplicar esses modelos fiz um tuning dos modelos antes. Onde temos um notebook para cada tuning dos modelos:

- [Árvore de decisão](https://github.com/barbosarafael/multiple-time-series-forecast/blob/main/02-notebooks/03-ml-models/02-tuning-dec-tree.ipynb)
- [Random Forest](https://github.com/barbosarafael/multiple-time-series-forecast/blob/main/02-notebooks/03-ml-models/03-tuning-random-forest.ipynb)
- [LGBM](https://github.com/barbosarafael/multiple-time-series-forecast/blob/main/02-notebooks/03-ml-models/04-tuning-lgbm.ipynb)
- [XGBoost](https://github.com/barbosarafael/multiple-time-series-forecast/blob/main/02-notebooks/03-ml-models/05-tuning-xgboost.ipynb)

E salvei os hiperparâmetros na pasta [03-best-params](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main/03-best-params). O processo de tuning dos modelos foi baseado no post do [Mario Filho](https://github.com/ledmaster), [Multiple Time Series Forecasting With LightGBM In Python](https://forecastegy.com/posts/multiple-time-series-forecasting-with-lightgbm-in-python/#tuning-lightgbm-hyperparameters-with-optuna).

Já que estamos falando de modelos de ML, podemos adicionar novas features para melhorar (ou não) o desempenho do modelo. Em séries temporais, as mais comuns são features que extraímos da data, como o dia da semana, dia do ano, semana do ano e etc, para identificar sazonalidade, tendência e padrões de datas anteriores. No `MLForecast` existe um parâmetro que você passa quais dessas features você deseja e ele mesmo extrai, em vez de ter que criar "na mão" com o Pandas. 

```python
date_features = ['dayofweek', 'month', 'year', 'quarter', 'day', 'week'] # Features de data
```

Outras features que podem ser extraídas são versões passadas da sua própria variável resposta (y), que são chamadas de *lag* ou *diff*. Que também são facilmente extraídas pelo `MLForecast`. 

```python
lags = [1, 7, 14, 21, 28, 30] # Criação de novas features de lags de 1, 7, ..., 30 dias da variável resposta
```

Podemos extrair também features de médias móveis, médias móveis sazonais, entre outras. Dependendo claro de quantos dias você vai querer que seja feito o cálculo da média móvel. O que é bem acessível de fazer com a ajuda das libs `window_ops` com o `numba`.

```python
#---- Features de data:

from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

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
```

Com isso, podemos aplicar todos esses parâmetros na nossa principal função, `MLForecast`, que fará o processo do `.fit`. 

```python
models_list = [lin_reg, dec_tree, tun_dec_tree, ran_forest, tun_ran_forest, lgbm, tun_lgbm, xgb, tun_xgb]

model = MLForecast(models = models_list, # Lista com 9 modelos
                   freq = 'D', # Frequência diária
                   num_threads = 6,
                   lags = [1, 7, 14, 21, 28, 30], # Criação de novas features de lags de 1, 7, ..., 30 dias da variável resposta
                   date_features = ['dayofweek', 'month', 'year', 'quarter', 'day', 'week'], # Features de data
                   lag_transforms = {
                       1: [expanding_mean],
                       7: [rolling_mean_7],
                       14: [rolling_mean_14],
                       21: [rolling_mean_21],
                       28: [rolling_mean_28],
                   }
           )

model.fit(Y_train_df.reset_index(), id_col = 'unique_id', time_col = 'ds', target_col = 'y', fitted = True)
```

Com o objeto `model` conseguimos acessar um método chamado `.preprocess` e visualizar um dataframe Pandas com o que foi treinado pelo modelo.

```python
model.preprocess(Y_train_df.reset_index())
```

![Alt text](04-images/image.png)

> Mais um adendo aqui. A nossa série inicia em 25/11/1997 mas no dataframe acima, os dados começam a ser treinados a partir do dia 19/01/1998. Não faz muito sentido. São 55 dias entre as duas datas e nenhum dos parâmetros que passei corresponde a esse valor 


### 4.4. Predict do modelo

#### Séries Temporais

```python
n_horizon = Y_valid_df.ds.nunique() # Quantidade de dias para a projeção

Y_hat_df = model.forecast(h = n_horizon, fitted = True)

Y_hat_df.head()
```

![Alt text](04-images/image-1.png)

A quantidade de dias para a projeção peguei diretamente da documentação, por isso o jeitinho diferente. O `fitted = True` vai servir para criarmos um novo dataframe com o que o modelo projetou para os dados de treino.

```python
Y_fitted_df = model.forecast_fitted_values()

display(Y_fitted_df.head())
display(Y_fitted_df.tail())
```

![Alt text](04-images/image-3.png)

#### Machine Learning


```python
Y_hat_df = model.predict(h = n_horizon)
```

O que difere entre as projeções é o método utilizado dentro `model`, um é `predict` ou outro é `forecast`.

![Alt text](04-images/image-4.png)

Dentro do `.fit` do modelo passamos um parâmetro de `fitted = True` também, com o mesmo objetivo dos modelos de Séries Temporais. E com o mesmo código temos o mesmo resultado.

![Alt text](04-images/image-6.png)

### 4.5. Reconciliação

Com a lib `hierarchicalforecast` facilmente podemos fazer o processo de reconciliação. Lembrando que esse é um processo pós-modelagem.

```python

from hierarchicalforecast.methods import BottomUp, TopDown, ERM, OptimalCombination, MinTrace, MiddleOut
from hierarchicalforecast.core import HierarchicalReconciliation

reconcilers = [BottomUp(), 
               TopDown(method = 'forecast_proportions'),
               TopDown(method = 'average_proportions'),
               TopDown(method = 'proportion_averages'),
               MiddleOut(middle_level = 'region/state', top_down_method = 'forecast_proportions'),
               MiddleOut(middle_level = 'region/state', top_down_method = 'average_proportions'),
               MiddleOut(middle_level = 'region/state', top_down_method = 'proportion_averages'),
               MinTrace(method = 'ols', nonnegative = True),
               MinTrace(method = 'wls_struct', nonnegative = True),
               MinTrace(method = 'wls_var', nonnegative = True),
               MinTrace(method = 'mint_shrink', nonnegative = True),
               OptimalCombination(method = 'ols', nonnegative = True),
               OptimalCombination(method = 'wls_struct', nonnegative = True)
              ]

hrec = HierarchicalReconciliation(reconcilers = reconcilers)

Y_rec_df = hrec.reconcile(Y_hat_df = Y_hat_df, # Dataframe com as projeções cruas
                          Y_df = Y_fitted_df, # Dataframe com as estimações do modelo nos dados de treino
                          S = S_df, # Sparse dataframe (0 ou 1) de cada combinação
                          tags = tags)

```

Pensando que estamos aqui para experimentar, vamos testar (quase) todos os métodos de reconciliação que tem disponíveis na lib. 

Primeiramente adicionamos em uma lista todos os métodos de reconciliação que queremos utilizar, logo após passamos eles para a função `HierarchicalReconciliation`. Essa parte de reconciliação é semelhante nos modelos de Séries Temporais e de Machine Learning. 

#### Séries Temporais

![Alt text](04-images/image-7.png)

#### Machine Learning

![Alt text](04-images/image-8.png)


Agora, com os métodos de reconciliação, temos 151 modelos diferentes que podemos analisar para identificar qual se aproxima mais das vendas realizadas. 

> Estava trocando ideia com um coordenador de dados da firma sobre esse projeto e ele fez a seguinte provocação: "O BottomUp se dá melhor nos menores níveis?". Não soube responder. Vendo os resultados aqui, vejo que não necessariamente.

## 5. Avaliação dos modelos

Agora que já nossas versões finais dos modelos, podemos efetivamente comparar o que eles projetaram com os nossos dados de vendas realizadas. 

Das métricas convencionais, iremos utilizar o RMSE. 

### 5.1. RMSE

![Alt text](https://miro.medium.com/v2/resize:fit:966/1*lqDsPkfXPGen32Uem1PTNg.png)

Métrica clássica de problemas de regressão dentro do Machine Learning e Estatística. A intuição por trás dele é: a média dos erros do seu modelo, em comparação com o que foi realizado. Onde a unidade é a mesma que a sua variável resposta.

Para o nosso contexto, isso significa que esses erros serão avaliados calculados nos dias dos nossos dados de validação, entre 01/01/2009 a 28/07/2009. 


```python

from hierarchicalforecast.evaluation import HierarchicalEvaluation

def rmse(y_true, y_pred):
    
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

evaluator = HierarchicalEvaluation(evaluators = [rmse])

evaluation = evaluator.evaluate(
    Y_hat_df = Y_rec_df,
    Y_test_df = Y_valid_df,
    tags = tags
)
```

![Alt text](04-images/image-9.png)

Com isso, o nosso objeto `evaluation` tem uma carinha parecida com o dataframe da imagem acima. Ele calcula o RMSE por nível e também geral para cada modelo. 

Para ficar mais fácil identificar quais os melhores modelos, i.e, os que tem menor RMSE, criei esses dois gráficos.

![Alt text](04-images/image-10.png)

No geral, o melhor modelo são os de Machine Learning, em específico a Regressão Linear, são os melhores, pois possuem menor RMSE, em comparação com a Baseline e os modelos de Séries Temporais. 

> E, como comentei anteriormente. As métricas entre os modelos ficam muito parecidas. 

### 5.2. Comparação gráfica

Uma comparação mais "olhométrica", iremos ver como as projeções estão se comportando em comparação com o que foi vendido. 

#### Séries Temporais

Aqui separei o resultado de um dos melhores modelos de Séries Temporais: AutoARIMA/BottomUp. Caso queira mudar o modelo, modifique o parâmetro `models`. 

As visualizações são dos maiores níveis, por default. Vemos que os resultados estão bem ruins, pois ele está basicamente projetando valores iguais, constantes. Logo, não é um bom modelo, apesar do RMSE.


```python
from utilsforecast.plotting import plot_series

plot_series(
    Y_train_df.reset_index().query('ds >= "2008-01-01"'), 
    Y_rec_df.reset_index().merge(Y_valid_df.reset_index(), on = ['unique_id', 'ds'], how = 'left'), 
    models = ['AutoARIMA/BottomUp'],
    plot_random = False, 
)
```

![Alt text](04-images/image-11.png)

Caso você queira especificar quais os níveis você queira ver, adicione o parâmetro `ids`. Exemplo: `ids = ['EastNorthCentral', 'EastNorthCentral/Illinois', 'EastNorthCentral/Illinois/womens_clothing']`.

#### Machine Learning

Vamos ver o desempenho do LGBMRegressor com o método de reconciliação OptimalCombination (WLS).

```python
plot_series(
    Y_train_df.reset_index().query('ds >= "2008-01-01"'), 
    Y_rec_df.reset_index().merge(Y_valid_df.reset_index(), on = ['unique_id', 'ds'], how = 'left'), 
    models = ['LGBMRegressor/OptimalCombination_method-wls_struct_nonnegative-True'],
    plot_random = False
)
```

![Alt text](04-images/image-12.png)

Temos uma variabilidade nas projeções, diferente dos modelos de Séries Temporais, mas precisaríamos olhar com mais calma as projeções dos demais. 

#### Opinião do autor

Os principais métodos já descrevi, talvez não tenhamos os melhores resultados. Para o seu contexto, caso siga esses passos, identifique quais são as combinações mais importantes, pode ser a que gera mais vendas, e verique como os modelos vem desempenhando nelas. 

## 6. Tabela final

Considerando que já temos um melhor modelo, junto a suas projeções. Como entregar essas projeções?

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/ef98a6d7-5d37-4037-b0f3-323f07b4bae1)

Acima temos os dados históricos que recebemos. Vamos criar uma tabela praticamente igual a essa, agora com as projeções. Vamos considerar que temos as projeções dos modelos de Machine Learning, já considerando  os métodos de reconciliação (`Y_rec_df`).

![Alt text](04-images/image-8.png)

A ideia aqui é apenar organizar o dataframe acima para que ele pareça com os dados que recebemos. Para isso, criei uma função, já aproveitando que iremos para um passo de automação no futuro.

```python
def create_final_df(df_pred: pd.DataFrame, cols_split: str):

    df1 = df_pred\
        .reset_index()\
        .assign(\
            nivel_hierarquia = lambda x: np.where(x['unique_id'].str.count('/') == 0, 1, x['unique_id'].str.count('/') + 1)
        )\
        .query('nivel_hierarquia == 3')
    
    df1[cols_split] = df1['unique_id'].str.split(pat = '/', n = len(cols_split), expand = True)
    
    df1 = df1[cols_split + ['ds'] + df1.select_dtypes(include = 'number').columns.tolist()]

    return df1

create_final_df(df_pred = Y_rec_df, cols_split = cols_hierarchical)
  
```

![Alt text](04-images/image-13.png)

O que fizemos aqui:

- reset_index: para que o index (`unique_id`) virasse uma coluna
- criação da variável `nivel_hierarquia` que identifica se o `unique_id` é da hierarquia 1, 2 ou 3 (1: nível região, 2: região/estado e 3: região/estado/item)
- **Filtramos somente os dados onde o nivel == 3**
- Abrimos a coluna de unique_id em 3: região, estado e item
- Selecionamos somente as colunas úteis

> Um adendo sobre o bullet em negrito. Selecionando somente o nível 3 estamos ignorando o restante dos níveis. Isso é uma boa prática? Confesso que não sei, pois não achei nenhum outro artigo em como entregar uma tabela final das projeções. Idealmente você tem que pesar entre os seus modelos nos diferentes níveis de projeção e ver o desempenho no nível 3, pois ele será o escolhido. Espero que essa não seja a única opção, pois é *paia*.

## 7. Bônus: Feature importance para os modelos de Machine Learning

No [artigo](https://mariofilho.com/como-prever-series-temporais-com-scikit-learn/#import%C3%A2ncia-das-features) do Mario Filho, ele ensina a extrair a feature importance dos modelos de Machine Learning. 

Vale lembrar aqui que cada modelo tem suas particularidades de feature importance. Por exemplo, o Random Forest mede a partir da impureza, quanto mais alto a impureza da variável, maior a importância (gini). 

Já o XGBoost tem 5 métodos diferentes para medir, depende do seu ponto de vista qual utilizar. Mas por padrão utiliza o `weight`, que rankeia as features a partir da quantidade de vezes que elas foram utilizadas para dividir os dados.

O código abaixo vai gerar a feature importance de todos os modelos, exceto da Regressão Linear. 

```python
import matplotlib.pyplot as plt

for mod in list(model.models_.keys()):

    if mod == 'LinearRegression':

        pass

    else:

        plt.figure(figsize = (10, 4))

        pd.Series(model.models_[mod].feature_importances_, 
                  index = model.ts.features_order_)\
            .sort_values(ascending = False)\
            .plot\
            .bar(title = f'{mod} Feature Importance',
                 xlabel = 'Features', 
                 ylabel = 'Importance')
```

![Alt text](04-images/image-14.png)

Um print da feature importance do LGBM. Na qual apontou que as features mais importantes foram a `lag1`, `dia da semana` e o `dia`.

## 8. Modelo em produção: 

**Link:** https://huggingface.co/spaces/barbosarafael/multiple-time-series-forecast. Caso o app não esteja ligado, pode liga-lo.  

Simulei adicionar esse modelo em produção, diretamente no Spaces do [Hugging Face](https://huggingface.co/spaces), juntamente com o `gradio`. Você pode conferir os arquivos na pasta [05-prd](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main/05-prd).

Temos 3 arquivos principais para fazer o deploy: `requirements.txt`, `functions.py` e `app.py`

#### requirements.txt

O requirements é um arquivo de texto com as bibliotecas e suas versões que utilizei nesse projeto. Ele é útil para você ter o exato mesmo ambiente de desenvolvimento que eu tinha quando estava desenvolvendo esse projeto. Tudo começa aqui, não pular para as outras etapas antes de instalar as libs.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/90177f59-2378-464b-a326-168444bb8a3a)

Para instalar essas bibliotecas de uma única vez, utilize: `pip install -r requirements.txt`.

#### functions.py

Contém todos os carregamentos das libs que utilizei e a criação das funções que serão aplicadas no script principal. Essas funções são:

1. Leitura dos dados
2. Limpeza dos dados
3. Organiza os dados de forma hierarquica
4. Aplica os modelos de séries temporais
5. Aplica os modelos de machine learning
6. Limpeza dos dados, após a aplicação dos modelos

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/aa1635be-d0dd-474a-9c81-98c3a888a6a0)

#### app.py

Script principal que carrega as funções do `functions.py` e cria um front-end simples para o deploy no Hugging Face. 

Inicialmente lemos os dados. Logo após, crio uma função de `predict`, onde o único argumento dessa função é a quantidade de dias para projeção. E output é um dataframe com as projeções.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/8cc782fa-470f-4870-9b41-89b96e1caa7e)


## 9. Referências

- **Hierarchical Forecasting in Python | Nixtla**: https://www.youtube.com/watch?v=lotzOJuwxYs&ab_channel=DataCouncil
- **Multiple Time Series Forecasting With LightGBM In Python**: https://forecastegy.com/posts/multiple-time-series-forecasting-with-lightgbm-in-python/
- **Como Prever Séries Temporais com Scikit-learn**: https://mariofilho.com/como-prever-series-temporais-com-scikit-learn/
- **Hierarchical TimeSeries Reconciliation**: https://medium.com/@adeforceville_96412/hierarchical-timeseries-reconciliation-58addce2aeb7
- **Chapter 11 Forecasting hierarchical and grouped time series**: https://otexts.com/fpp3/hierarchical.html
- **Nixtla/hierarchicalforecast**: https://github.com/Nixtla/hierarchicalforecast


## 10. Melhorias e aprendizados: 

- Extrair features de feriados: isso ainda não é possível pela lib
- Adicionar os intervalos de confiança
- Modelar com alguma transformação: como box-cox ou raiz quadrada
- Variáveis com muitas categorias são extremamente detratoras no passo de Reconciliação: computacionalmente custoso.
