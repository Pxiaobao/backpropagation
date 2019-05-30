import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initParam(network_shape):
    parameters = {}
    for i in range(1, len(network_shape)):
        parameters['w'+ str(i)] = np.random.random((network_shape[i]*network_shape[i-1],1))
        parameters['b'+ str(i)] = np.zeros((network_shape[i],1))
    return parameters
if __name__ == '__main__':
    parameters = initParam([1,25,1])
    x = np.arange(0,1,0.01)
    y = 10*np.sin(2*np.pi*x)
    x = x.reshape(1, 100)
    # plt.scatter(x, y)
    # plt.title("函数曲线", fontsize=24)
    # plt.xlabel("X", fontsize=14)
    # plt.ylabel("Y", fontsize=14)
    # plt.show()
    result1 = parameters['w1'].dot(x)+parameters['b1']
    print(result1)
    #print(parameters)