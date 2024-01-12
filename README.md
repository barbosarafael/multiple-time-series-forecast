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

## 3. Resultados

### 3.1. Exploração

Apenas alguns highlights:

- As datas mais recentes são as que possuem mais vendas
- Tennessee (1.351.608), Kentucky (1.298.425) e California (1.173.635) são os estados com mais vendas
- Vestimentas de mulheres são os itens mais vendidos (11.283.595)
- SouthCentral é a região/regional com maior quantidade de vendas (4.792.847)
- Tennessee/womens_clothing/SouthCentral foi a combinação (variáveis region/state/item) que teve mais vendas durante o período 
- Tendência nas no aumento das vendas no varejo, perceptível a partir dos anos 2000 (ver imagem abaixo)

![image](https://github.com/barbosarafael/multiple-time-series-forecast/assets/44044829/5bb81ca3-3402-4207-825e-9f7a137b3da5)

Dentro da pasta [01-explorations](https://github.com/barbosarafael/multiple-time-series-forecast/tree/main/02-notebooks/01-explorations) tem as análises mais detalhadas. 

### 3.2. Modelos

- resultado das métricas
- projeções no tempo
- tunagem dos modelos com optuna
- dataframe com o que foi passado para o modelo (ML)
- feature importance

## 4. Modelo em produção

## Referências
