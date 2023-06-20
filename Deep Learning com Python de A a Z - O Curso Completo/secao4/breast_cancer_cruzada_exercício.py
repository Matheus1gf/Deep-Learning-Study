import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=16, activation='sigmoid', kernel_initializer='random_uniform', input_dim=30))
    
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=16, activation='sigmoid', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2))
       
    classificador.add(Dense(units=1, activation='sigmoid'))
    print(classificador)
    print("-------------------------------------------------------------------------")

    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    print("-------------------------------------------------------------------------")

    classificador.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['binary_accuracy'])
    print("-------------------------------------------------------------------------")
    return classificador

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

classificador = KerasClassifier(build_fn=criarRede, epochs=150, batch_size=50)
print(classificador)
print("-------------------------------------------------------------------------")

resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')
print(resultados)
print("-------------------------------------------------------------------------")

media = resultados.mean()
print(media)
print("-------------------------------------------------------------------------")

desvio = resultados.std()
print(desvio)
print("-------------------------------------------------------------------------")