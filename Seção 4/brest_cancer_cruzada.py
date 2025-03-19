import pandas as pd
import sklearn
import tensorflow as tf
import scikeras
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import backend as k

def creat_net():
    k.clear_session()
    neural_network = Sequential([
        tf.keras.layers.Input(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    neural_network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='binary_accuracy')
    return neural_network

X = pd.read_csv('./data/entradas_breast.csv')
y = pd.read_csv('./data/saidas_breast.csv')

neural_network = KerasClassifier(model=creat_net, epochs=100, batch_size=10)
results = cross_val_score(estimator=neural_network, X=X, y=y, cv=10, scoring='accuracy')
print(results)
print("-----------------------------------------------------------------------------------------------")

print(results.mean())
print(results.std())