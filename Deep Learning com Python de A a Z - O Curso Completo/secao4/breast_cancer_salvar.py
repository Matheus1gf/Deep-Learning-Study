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

# Criando arquivo JSON para salvar
classificador_json = classificador.to_json()

# Salvando arquivo
with open(origem+"secao4/classificador_breast.json", 'w') as json_file:
    json_file.write(classificador_json)

# Salvando pesos
classificador.save_weights(origem+"secao4/classificador_breast.h5")