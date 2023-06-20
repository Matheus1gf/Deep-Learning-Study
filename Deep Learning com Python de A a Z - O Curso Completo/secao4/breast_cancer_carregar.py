import numpy as np
from keras.models import model_from_json
import pandas as pd

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"

# Chamando o arquivo JSON salvo e lendo ele com 'r'
arquivo = open(origem+"secao4/classificador_breast.json", 'r')
# Lendo a estrutura da rede neural
estrutura_rede = arquivo.read()
# Fechando arquivo para liberar memória
arquivo.close()

# Definindo o classificador
classificador = model_from_json(estrutura_rede)

# Carregando os pesos
classificador.load_weights(origem+"secao4/classificador_breast.h5")

# Criando os novos dados a serem tratados
# Obs: é posto dois conchetes para criar o array pois o primeiro é a linha e o segundo são os preenchimentos das colunas daquela linha
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
print(novo)
print("-------------------------------------------------------------------------")

# Efetuando a previsao em cima dos novos dados
previsao = classificador.predict(novo)
print(previsao)
print("-------------------------------------------------------------------------")

# Puxando valor 'true' ou 'false'
previsao = (previsao > 0.5)
print(previsao)
print("-------------------------------------------------------------------------")

previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

# Fazendo avaliação utilizando a rede neural carregada
classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)
print(resultado)