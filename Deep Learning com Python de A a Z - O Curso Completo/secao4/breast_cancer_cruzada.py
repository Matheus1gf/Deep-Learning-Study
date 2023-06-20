import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def criarRede():
    classificador = Sequential()
    # Com o Dense() iremos startar as camadas ocultas da rede neural
    # Units = número de entradas + saídas dividido por 2
    # Actvation = função de ativação
    # Kernel_initializer = inicialização dos pesos
    # Input_dim = Quantos elementos existem na camada de entradqa
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    
    # Adicionando uma camada de dropout para evitar o overfiting
    # Ele pegará 20% dos neurônios de entrada e irá zerar, com isso teremos uma melhoria de +/- 5%
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2))
       
    # Fazendo a camada de saída
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
    return classificador

origem = "C:/Users/mathe/Udemy/Deep Learning com Python de A a Z - O Curso Completo/"
previsores = pd.read_csv(origem+"secao4/entradas_breast.csv")
classe = pd.read_csv(origem+'secao4/saidas_breast.csv')
print(previsores)
print(classe)
print("-------------------------------------------------------------------------")

# build_fn = responsável pela criação da rede neural
# epochs = quantidade de épocas
classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)
print(classificador)
print("-------------------------------------------------------------------------")

# estimador = classificador da rede
# X = previsores
# y = classe da rede
# cv = quantidade de execuções (divisão da base de dados)
# scoring = retorno desejado
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')
print(resultados)
print("-------------------------------------------------------------------------")

# Para efetivmente saber a quantidade de acertos da validação cruzado acima é necessário se fazer uma média
media = resultados.mean()
print(media)
print("-------------------------------------------------------------------------")

# Calculando desvio padrão para saber qual a variação dos valores
desvio = resultados.std()
print(desvio)
print("-------------------------------------------------------------------------")