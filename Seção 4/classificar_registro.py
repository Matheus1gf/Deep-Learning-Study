import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout

X = pd.read_csv('./data/entradas_breast.csv')
y = pd.read_csv('./data/saidas_breast.csv')

classificador = Sequential()
classificador.add(tf.keras.Input(shape=(30,)))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(X, y, batch_size=10, epochs=100)
print(X)
print("-----------------------------------------------------------------------------------------------")

novo = np.array([[ 15.80,  8.34,   118,   900,  0.10,  0.26,  0.08, 0.134, 0.178,  0.20,
                    0.05,  1098,  0.87,  4500, 145.2, 0.005,  0.04,  0.05, 0.015,  0.03,
                   0.007, 23.15, 16.64, 178.5,  2018,  0.14, 0.185,  0.84,   158, 0.363]])
print(novo)
print("-----------------------------------------------------------------------------------------------")

print(novo.size, novo.shape)
print("-----------------------------------------------------------------------------------------------")

previsao = classificador.predict(novo)
print(previsao)
print("-----------------------------------------------------------------------------------------------")

previsao = previsao > 0.5
print(previsao)
print("-----------------------------------------------------------------------------------------------")

if previsao:
  print('Malígno')
else:
  print('Benigno')

classificador.save('classificador_breast.keras')
classificador_novo = tf.keras.models.load_model('./classificador_breast.keras')
classificador_novo.summary()
classificador_novo.predict(novo)