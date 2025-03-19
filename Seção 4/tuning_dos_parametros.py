import pandas as pd
import tensorflow as tf
import sklearn
import scikeras
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV

def creat_net(optimizer, loss, kernel_initializer, activation, neurons):
  k.clear_session()
  neural_network = Sequential([
      tf.keras.Input(shape=(30,)),
      tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dropout(rate = 0.2),
      tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
  neural_network.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
  return neural_network

X = pd.read_csv('./data/entradas_breast.csv')
y = pd.read_csv('./data/saidas_breast.csv')

neural_network = KerasClassifier(model = creat_net)
parameters = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8]
}
print(parameters)
print("-----------------------------------------------------------------------------------------------")

grid_search =  GridSearchCV(estimator=neural_network, param_grid=parameters, scoring='accuracy', cv=5)
grid_search = grid_search.fit(X, y)
print(grid_search)
print("-----------------------------------------------------------------------------------------------")

best_params = grid_search.best_params_
print(best_params)
print("-----------------------------------------------------------------------------------------------")

best_precision = grid_search.best_score_
print(best_precision)