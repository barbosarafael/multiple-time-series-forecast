# Projeção de múltiplas séries temporais

Salve! Nesse repositório irei mostrar como podemos fazer projeções múltiplas séries temporais de uma única vez. 

## 1. Dados utilizados

Os dados se referem a venda de roupa no varejo dos USA. Os dados raw podem ser encontrados diretamente nesse [link](https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-hierarchical-forecasting/main/retail-usa-clothing.csv). Abaixo um print da tabela.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/ef98a6d7-5d37-4037-b0f3-323f07b4bae1)

**Nosso objetivo é projetar as vendas de roupas no varejo de todas as "combinações/concatenação" das 3 variáveis do dataset: state, item e region**. 

*Obs.: A variável **country** sempre será USA, por isso não iremos considerá-la no estudo.*

## 2. Projeções hierárquicas (hierarchical forecast)

### 2.1. Situação hipotética

Pensa que tu trabalhas nessa empresa de varejo e o seu chefe pediu para você fazer a projeção de quantas vendas terão para os itens que vocês vendem (roupas feminas e masculinos, sapatos, etc...). Você deve imaginar, ok, são apenas 5 ou 6 itens para projetar, dá para modelar cada série individualmente, na mão. 

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/cdc477c3-1814-4339-86c7-fabc050d8b60)

**Mas** aí você encontra sua primeira dificuldade: as lojas estão espalhadas pelos estados do USA. Logo, agora você não tem somente os 6 itens para modelar e sim os 6 itens dentro de cada estado no banco.

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/c963677b-f4b0-4bbf-9bae-948df13d2d9e)

Agora teremos que projetar 108 (6 itens $\times$ 18 estados) séries temporais diferentes. Já pensa ter que modelar uma a uma? Na mão? Trabalhoso né?

> Adendo e provocação: Você pode aplicar métodos automáticos de projeção para cada série, como um AutoARIMA da vida. Certíssimo, eu faria isso. Mas e se o seu chefe te pedir para agregar as projeções somente do estado de Nova York? Será que, quando você fazer a agregação para o nível de estado, as projeções irão ser exatamente iguais?

**Entrando** aí a sua segunda (e última) dificuldade. Não basta ter que projetar as vendas dos itens dentro de cada Estado. Cada estado, vai estar dentro de uma região... Logo, você tem mais um nível de projeção, mas esse não é tão dificultoso assim...

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/7dd5ee75-0666-44f4-8c64-6ffa1650a834)

O lado positivo dessa "última dificuldade" é que um Estado só pode estar dentro de uma Região. Logo, não teremos novas combinações. 

**Para** o nosso caso, temos mais um lado positivo. O gráfico abaixo mostra quantas combinações distintas entre as 3 variáveis foram vendidas durante o tempo, ou seja, quantas projeções de vendas teremos que fazer diariamente. Sempre foram 91 combinações e não aquelas 108 que havia comentado anteriormente. Em termos práticos, em alguns estados não são vendidos determinados itens.

![teste](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/cb813600-7aef-4a8a-b0e4-4eefa698a039)

### 2.2. Um pouquinho de teoria (na prática)

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/8211ebe8-27c0-41e2-b33a-8a70da774157)

De forma visual e resumida, a nossa hierarquia de variáveis pode ser representada da maneira acima.

Agora vamos supor que **rodamos modelos** para projetar as vendas para as séries, dentro de cada nível, isto é, vamos fazer as projeções para cada um dos valores que temos dentro dos níveis. Exemplos abaixo:

- Nível I: SouthCentral
- Nível II: SouthCentral/Vermont, SouthCentral/Maine e SouthCentral/Connecticut
- Nivel III:
  - SouthCentral/Vermont/Roupas Femininas, SouthCentral/Vermont/Roupas Masculinas e SouthCentral/Vermont/Roupas Infantil
  - SouthCentral/Maine/Roupas Femininas, SouthCentral/Maine/Roupas Masculinas e SouthCentral/Maine/Roupas Infantil
  - SouthCentral/Connecticut/Roupas Femininas, SouthCentral/Connecticut/Roupas Masculinas e SouthCentral/Connecticut/Roupas Infantil
 
Infelizmente, na hora de agregar os valores nos níveis acima os valores não batem. Para o nosso caso, estamos falando que a projeção das vendas de todos os Estados na Região SouthCentral não batem com as vendas da Região SouthCentral.

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

Como comentei acima, a Reconciliação é o método que vai fazer a soma das projeções dos diferentes níveis baterem. Ele é um processo após a projeção, como podemos ver na imagem abaixo.

<p>
    <img src="https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/1a2c4a23-e417-44cf-ad33-c4efc5c35a26" alt>
    <em>Retirado de: Hierarchical TimeSeries Reconciliation by Adrien</em>
</p>

Existem uma boa quantidade de métodos de reconciliação. Os principais são:

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

Existem outros métodos que mudam a forma a estratégia de como esse processo de "reestimação" vai acontecer e outras focadas em diminuir o erro. Algumas delas são:

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
- Tendência nas no aumento das vendas no varejo, perceptível a partir dos anos 2000 (ver imagem abaixo)

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/5bb81ca3-3402-4207-825e-9f7a137b3da5)

Dentro da pasta [01-explorations](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main/02-notebooks/01-explorations) tem as análises mais detalhadas. 

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

A segunda modificação é o formato dos dados. Para isso, vamos utilizar a função aggregate. Ela exige dois parâmetros: um dataframe com os dados (ver print acima) e uma lista de lista das hierarquias `[['region'], ['region', 'state'], ['region', 'state', 'item']]`. 

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

- Treino: 25/11/1997 a 31/12/2008
- Validação: 01/01/2009 a 28/07/2009 (quase 8 meses)

Teremos um **Y_train_df** e um **Y_valid_df**.

### 4.3. Aplicação dos modelo 

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

O processo de tuning dos modelos foi baseado no post do Mario Filho, [Multiple Time Series Forecasting With LightGBM In Python](https://forecastegy.com/posts/multiple-time-series-forecasting-with-lightgbm-in-python/#tuning-lightgbm-hyperparameters-with-optuna).


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

> Um adendo aqui, na documentação da lib, eles aceitam dataframes do Pandas, Spark, Dask e Ray. Li em algum local que estão começando a desenvolver para aceitar em Polars também.

#### Machine Learning




- Falar dos modelos tunados
- Mostrar o que foi "treinado" preprocess

### 4.4. Predict do modelo

### 4.5. Reconciliação

## 5. Avaliação dos modelos

### 5.1. RMSE

### 5.2. Comparação gráfica

## 6. Tabela final

## 7. Feature importance

## 8. Modelo em produção

## 9. Referências
