import numpy as np

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

def tanFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

testeStep = stepFunction(30)
testeStep2 = stepFunction(-1)

testeSigmoid = sigmoidFunction(2.1)
testeSigmoid2 = sigmoidFunction(1)
testeSigmoid3 = sigmoidFunction(0.358)

testeTan = tanFunction(2.1)
testeTan2 = tanFunction(-0.358)

testeRelu = reluFunction(2.1)
testeRelu2 = reluFunction(1)
testeRelu3 = reluFunction(2)

testeLinear = linearFunction(2.1)

valores = [5.0, 2.0, 1.3]
testeSoftmax = softmaxFunction(valores)

print(testeStep)
print(testeStep2)
print('-----------------')

print(testeSigmoid)
print(testeSigmoid2)
print(testeSigmoid3)
print('-----------------')

print(testeTan)
print(testeTan2)
print('-----------------')

print(testeRelu)
print(testeRelu2)
print(testeRelu3)
print('-----------------')

print(testeLinear)
print('-----------------')

print(testeSoftmax)
print('-----------------')

print(testeSigmoid)
print(testeTan)
print(testeRelu)
print(testeLinear)
print(testeSoftmax)
