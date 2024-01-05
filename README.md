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

### 2.2. Um pouquinho de teoria

- reconciliação
- tipos de reconciliação: bottomup, topdown e demais

## 3. Resultados

### 3.1. Exploração

### 3.2. Modelos

- resultado das métricas
- projeções no tempo
- tunagem dos modelos com optuna
- dataframe com o que foi passado para o modelo (ML)
- feature importance

## 4. Modelo em produção

## Referências
