import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

base = pd.read_csv("./data/iris.csv")
print(base)
print("-----------------------------------------------------------------------------------------------")

X = base.iloc[:, 0:4].values
print(X)
print("-----------------------------------------------------------------------------------------------")

y = base.iloc[:, 4].values
print(y)
print("-----------------------------------------------------------------------------------------------")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)
print(y.shape)
print("-----------------------------------------------------------------------------------------------")

y = np_utils.to_categorical(y)
print(y)
print("-----------------------------------------------------------------------------------------------")

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25)
print(X_treinamento.shape)
print(X_teste.shape)
print(y_treinamento.shape)
print(y_teste.shape)
print("-----------------------------------------------------------------------------------------------")

rede_neural = Sequential([
    tf.keras.layers.Input(shape = (4,)),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 3, activation = 'softmax')
])

print(rede_neural.summary())
print("-----------------------------------------------------------------------------------------------")

rede_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
rede_neural.fit(X_treinamento, y_treinamento, batch_size = 10, epochs = 1000)
rede_neural.evaluate(X_teste, y_teste)
previsoes = rede_neural.predict(X_teste)
print(previsoes)
print("-----------------------------------------------------------------------------------------------")

previsoes = previsoes > 0.5
print(previsoes)
print("-----------------------------------------------------------------------------------------------")

y_teste2 = [np.argmax(t) for t in y_teste]
print(y_teste2)
print("-----------------------------------------------------------------------------------------------")

previsoes2 = [np.argmax(t) for t in previsoes]
print(previsoes2)
print("-----------------------------------------------------------------------------------------------")

print(accuracy_score(y_teste2, previsoes2))
print("-----------------------------------------------------------------------------------------------")

print(confusion_matrix(y_teste2, previsoes2))
print("-----------------------------------------------------------------------------------------------")