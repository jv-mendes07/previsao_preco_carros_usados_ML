# Previsão de Preço de Carros Usados (Projeto de Regressão)
## Projeto de Machine Learning

Neste projeto de dados, utilizei um conjunto de dados relativo ao preço de venda de vários carros usados na Índia, e com base em tal conjunto de dados, construi um modelo preditivo de regressão linear para prever o preço de venda de tais carros usados.

Tal projeto é dividido entre às fases de **(1)** tratamento de dados, **(2)** análise exploratória de dados, **(3)** preparação e treino do modelo de regressão linear e **(4)** avaliação do modelo final.

Após a importação do dataset, verifiquei o formato de linhas e colunas do conjunto de dados:

* 301 linhas
* 9 colunas

Em seguida, visualizei às primeiras cinco linhas da tabela:

|          |Car_Name | Year    | Selling_Price | Present_Price  | Kms_Driven| Fuel_Type   | Seller_Type  | Transmission  |Owner         |
|----------|---------|---------|---------------|----------------|-----------|-------------|--------------|---------------|--------------|
| 0        | ritz    | 2014    | 3.35          | 5.59           | 27000     | Petrol      | Dealer       | Manual        | 0            |
| 1        | sx4     | 2013    | 4.75          | 9.54           | 43000     | Diesel      | Dealer       | Manual        | 0            |
| 2        | ciaz    | 2017    | 7.25          | 9.85           | 6900      | Petrol      | Dealer       | Manual        | 0            |
| 3        | wagon r | 2011    | 2.85          | 4.15           | 5200      | Petrol      | Dealer       | Manual        | 0            |
| 4        | swift   | 2014    | 4.60          | 6.87           | 42450     | Diesel      | Dealer       | Manual        | 0            |

## Tratamento de dados:

Na fase de tratamento de dados, realizei:

* **Reformatação textual do nome das colunas:**

Construi um list-compreehension para converter todos os nomes das colunas do dataset em letras minúsculas, desse modo, quando fosse me referir ao nome das colunas, não teria que escrever com letra maiúscula a primeira letra de cada coluna:

```
# Formatação textual do nome das colunas, para que todas colunas estejam em minúsculo:

df.columns = [x.lower() for x in df.columns]
```
Saída com o nome das colunas transformado:

```
Index(['car_name', 'year', 'selling_price', 'present_price', 'kms_driven',
       'fuel_type', 'seller_type', 'transmission', 'owner'],
      dtype='object') 
 ```
* **Tratamento de dados nulos:**

Usei o método .isnull().sum() para verificar a quantidade de dados nulos em cada coluna no conjunto de dados e obtive:

```
car_name         0
year             0
selling_price    0
present_price    0
kms_driven       0
fuel_type        0
seller_type      0
transmission     0
owner            0
dtype: int64
```

Ou seja, não havia nenhum dado ausente registrado no conjunto de dados.

Após concluir os principais passos da limpeza de dados, decidi começar o processo de análise exploratória dos dados, para extrair informações e insights importantes sobre o conjunto de dados em questão:

## Análise Exploratória De Dados (EDA):

Antes de inicializar a etapa de análise exploratória, é imprescindível ter um dicionário de dados que explique rapidamente o quê cada coluna informa no conjunto de dados que será analisado:

#### Dicionário de Dados

* **name** - Nome do carro
* **year** - Ano de compra do carro
* **selling_price** - Preço de venda do carro
* **present_price** - Preço atual do carro na concessionária
* **kms_driven** - Distância em kilomêtros percorrida pelo carro
* **fuel_type** - Tipo de combustível do carro
* **seller_type** - Tipo de vendedor (indivíduo ou distribuidora)
* **transmission** - Se o carro é manual ou automático
* **owner** - Quantidade de donos que tal carro já teve anteriormente


