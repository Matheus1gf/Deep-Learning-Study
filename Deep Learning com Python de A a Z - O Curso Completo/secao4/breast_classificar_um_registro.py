import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

classificador = Sequential()
classificador.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))
print(classificador)
print("-------------------------------------------------------------------------")

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
print(classificador)
print("-------------------------------------------------------------------------")

classificador.fit(previsores, classe, batch_size=10, epochs=100)
print(classificador)
print("-------------------------------------------------------------------------")

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