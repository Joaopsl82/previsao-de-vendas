# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos
# - Passo 7: Interpretação de Resultados

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv('advertising.csv')

# Visualizar como as informações de cada item estão distribuídas
sns.heatmap(tabela.corr(), annot=True, cmap='Wistia')
plt.show()


# Separando em dados de treino e dados de teste
y = tabela['Vendas']
x = tabela.drop('Vendas', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# Cria as IA
modelo_regrassaoLinear = LinearRegression()
modelo_arvoreDecisao = RandomForestRegressor()

# Treinar as IA
modelo_regrassaoLinear.fit(x_treino, y_treino)
modelo_arvoreDecisao.fit(x_treino, y_treino)

# Criar as Previsões
previsao_regressaoLinear = modelo_regrassaoLinear.predict(x_teste)
previsao_arvoreDecisao = modelo_arvoreDecisao.predict(x_teste)

# Comparar os Modelos
print(metrics.r2_score(y_teste, previsao_regressaoLinear))
print(metrics.r2_score(y_teste, previsao_arvoreDecisao))

# Visualização gráfica das Previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsões ArvoreDecisao'] = previsao_arvoreDecisao
tabela_auxiliar['Previsões Regressão Linear'] = previsao_regressaoLinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

# Fazer nova previsão
nova_tabela = pd.read_csv('novos.csv')
print(nova_tabela)
previsao = modelo_arvoreDecisao.predict(nova_tabela)
print(previsao)

# Atribuindo cada variável para vendas
sns.barplot(x=x_treino.columns, y=modelo_arvoreDecisao.feature_importances_)
plt.show()