import pandas as pd
from sklearn.model_selection import train_test_split

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
base = pd.read_csv(origem+"secao4/entradas_breast.csv")
print(base)
print("-------------------------------------------------------------------------")

# Dividindo a base em intervalor de 0 a 4 com todas as linhas e utilizando .value para converter para varável numpy
previsores = base.iloc[:, 0:4].values
print(previsores)
print("-------------------------------------------------------------------------")

# Dividindo a base em intervalor de 4 adiante com todas as linhas e utilizando .value para converter para varável numpy
classe = base.iloc[:, 4].values
print(classe)
print("-------------------------------------------------------------------------")

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.36)
print(previsores_treinamento)
print(previsores_teste)
print(classe_treinamento)
print(classe_teste)
print("-------------------------------------------------------------------------")