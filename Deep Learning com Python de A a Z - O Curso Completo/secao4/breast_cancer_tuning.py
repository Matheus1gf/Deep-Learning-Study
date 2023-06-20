import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=1, activation='sigmoid'))
    print(classificador)
    print("-------------------------------------------------------------------------")

    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    print(classificador)
    print("-------------------------------------------------------------------------")
    return classificador

# Chamando a funçaõ criarRede dinâmicamente
classificador = KerasClassifier(build_fn=criarRede)
print(classificador)
print("-------------------------------------------------------------------------")

# Criando parâmetros para se utilizar na função criarRede
parametros = {
    'batch_size' : [10, 30, 50],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop', 'adamax'],
    'loss': ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'],
    'kernel_initializer': ['random_uniform', 'normal', 'zeros'],
    'activation': ['relu', 'sigmoid', 'softmax'],
    'neurons': [32, 16, 8]
    }
print(parametros)
print("-------------------------------------------------------------------------")

# Criando grid search para efetivamente realizar a busca
# estimator = classificador
# param_grid = grade de parâmetros
# socring = como é feita a avaliação dos resultados
# cv = números de folds
grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)
print(grid_search)
print("-------------------------------------------------------------------------")

grid_search = grid_search.fit(previsores, classe)
print(grid_search)
print("-------------------------------------------------------------------------")

# Retorna uma lista com os melhores parâmetros a serem passados
melhores_parametros = grid_search.best_params_
print(melhores_parametros)
print("-------------------------------------------------------------------------")

# Retorna uma lista com os melhores dados a serem passados
melhor_precisao = grid_search.best_score_
print(melhor_precisao)
print("-------------------------------------------------------------------------")