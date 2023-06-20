import pandas as pd
# https://keras.io
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)
print(previsores_treinamento)
print(previsores_teste)
print(classe_treinamento)
print(classe_teste)
print("-------------------------------------------------------------------------")

classificador = Sequential()
# Com o Dens() iremos startar as camadas ocultas da rede neural
# Units = número de entradas + saídas dividido por 2
# Actvation = função de ativação
# Kernel_initializer = inicialização dos pesos
# Input_dim = Quantos elementos existem na camada de entradqa
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
# Fazendo a camada de saíde
# Units = 1 pois só há uma camada de saída que informará se o tumor é malígno ou benígno (0 e 1)
# Activation = sigmoid pois a rede só retorna 0 e 1
classificador.add(Dense(units=1, activation='sigmoid'))
print(classificador)
print("-------------------------------------------------------------------------")

# Compilando a rede neural
# Optimizer = qual a função utilizada para fazer o ajuste dos pesos
# Loss = função de perda (utilizada binary_crossentropy para saída binária)
# Metrics = quais a métrica para a função
#classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
#print("-------------------------------------------------------------------------")

# Configurando alguns parâmetros do adam
# https://keras.io/api/optimizers/
# lr = learning rate (taxa de aprendizagem) utilizado para chegar no mínimo global
# decay = define de quanto em quanto será o decaimento da aproximação do mínimo global
# clipvalue = normaliza a busca do minímo global, evitando um loop na definição do número
otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
print("-------------------------------------------------------------------------")

# Executando o classificador com as novas parametrizações acima
# Esta execução é diferente da execução da linha 42 pois lá os parâmetros eram padrões do Adam,
# já nesta execução abaixo estamos executando o otimizador com os parâmetros personalizados
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
print("-------------------------------------------------------------------------")

# Startando o treinamento
# batch_size = vai calcular o erro para X registros e atualizar os pesos
# Epochs = Quantas épocas (vezes) serão feitos os ajustes dos pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)
print("-------------------------------------------------------------------------")

# Com layers[0] é selecionada a primeira camada da rede neural
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
print("-------------------------------------------------------------------------")

pesos1 = classificador.layers[1].get_weights()
print(pesos1)
print(len(pesos1))
print("-------------------------------------------------------------------------")

pesos1 = classificador.layers[2].get_weights()
print(pesos1)
print(len(pesos1))
print("-------------------------------------------------------------------------")

# Startando o teste
previsoes = classificador.predict(previsores_teste)
# Tratando para retornar "true" ou "false"
previsoes = (previsoes > 0.5)
print(previsoes)
print("-------------------------------------------------------------------------")

# Fazendo o comparativos entre dois vetores para identificar o que está correto
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)
print("-------------------------------------------------------------------------")

# Criando a matriz de precisão
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)
print("-------------------------------------------------------------------------")

# Submetendo os dados para a rede neural para fazer os comparativos no Keras
resultado = classificador.evaluate(previsores_teste, classe_teste)
print(resultado)