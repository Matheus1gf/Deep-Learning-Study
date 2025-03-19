import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix

# previsores
X = pd.read_csv('./data/entradas_breast.csv')

# classe
y = pd.read_csv('./data/saidas_breast.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print("-----------------------------------------------------------------------------------------------")

neural_network = Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'), # (30+1)/2
    tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Exibindo o resumo do modelo
neural_network.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)

neural_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='binary_accuracy')
neural_network.fit(X_train, y_train, batch_size=10, epochs=100)

pesos0 = neural_network.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
print(pesos0[1])
print("-----------------------------------------------------------------------------------------------")

pesos1 = neural_network.layers[1].get_weights()
print(pesos1)
print(len(pesos1))
print(pesos1[1])
print("-----------------------------------------------------------------------------------------------")

pesos2 = neural_network.layers[2].get_weights()
print(pesos2)
print(len(pesos2))
print(pesos2[1])
print("-----------------------------------------------------------------------------------------------")

predict = neural_network.predict(X_test)
predict = predict > 0.5
print(predict)
print(y_test)
print("-----------------------------------------------------------------------------------------------")

print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict))
print(neural_network.evaluate(X_test, y_test))
