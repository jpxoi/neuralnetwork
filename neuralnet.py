#Basic Libraries Import
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

#Dataset
print("Original Dataset")

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

Y = Y[:, np.newaxis]

plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="blue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="red")
plt.axis("equal")
plt.show()

#Neural Network
  #Neural Layers
  class neural_layer():
  def __init__(self, n_conn, n_neur, act_f):
    self.act_f = act_f
    self.b = np.random.rand(1, n_neur) * 2 - 1
    self.W = np.random.rand(n_conn, n_neur) * 2 - 1
  
  #Activation Function
    #Sigmoid Function
      #Function
      sigm = (
        lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x)
        )

      #Variables
      rang = np.linspace(-5, 5, 100)
      sigm_func = sigm[0](rang)
      sigm_derv = sigm[1](rang)

      #Graph Plotting
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
      axes[0].plot(rang, sigm_func)
      axes[1].plot(rang, sigm_derv)
      fig.tight_layout()
      
    #ReLu Function
      #Function
      def relu_derivate(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

      relu = (
        lambda x: x * (x > 0),
        lambda x: relu_derivate(x)
        )

      #Variables
      rang = np.linspace(-5, 5, 100)
      relu_func = relu[0](rang)
      relu_derv = relu[1](rang)

      # Graph Plotting
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
      axes[0].plot(rang, relu_func)
      axes[1].plot(rang, relu_derv)
      plt.show()
      
  #Neural Network
    #Creation
    def create_nn(topology, act_f):

      nn = []

      for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))

      return nn

    topology = [p, 4, 8, 4, 1]

    neural_network = create_nn(topology, sigm)

    print(neural_network)
    
    #Training
      #Cost Function
      l2_cost = (
        lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
        lambda Yp, Yr: (Yp - Yr)
      )
      
      #Training Process
      def train(neural_network, X, Y, l2_cost, lr=0.5, train=True):

        out = [(None, X)]

        #Forward Pass
        for l, layer in enumerate(neural_network):
          z = out[-1][1] @ neural_network[l].W + neural_network[l].b
          a = neural_network[l].act_f[0](z)

          out.append((z, a))

        if train:

          #Backward Pass
          deltas = []

          for l in reversed(range(0, len(neural_network))):

            z = out[l+1][0]
            a = out[l+1][1]

            if l == len (neural_network) -1:
              deltas.insert(0, l2_cost[1](a, Y) * neural_network[l].act_f[1](a))

            else:
              deltas.insert(0, deltas[0] @ _W.T * neural_network[l].act_f[1](a))

            _W = neural_network[l].W

            #Gradient Descent
            neural_network[l].b = neural_network[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_network[l].W = neural_network[l].W - out[l][1].T @ deltas[0] * lr

        return out[-1][1]

      train(neural_network, X, Y, l2_cost, 0.5)
      
#Runing The Neuronal Network
import time
from IPython.display import clear_output

neural_n = create_nn(topology, sigm)

loss = []

for i in range (2500):

  #Training
  pY = train(neural_n, X, Y, l2_cost, lr=0.051)

  if i % 25 == 0:

    print (pY)

    loss.append (l2_cost[0](pY, Y))

    res = 50
    
    _x0 = np.linspace(-1.5, 1.5, res)
    _x1 = np.linspace(-1.5, 1.5, res)

    _Y = np.zeros((res, res))

    for i0, x0 in enumerate(_x0):
      for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]

    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
    plt.axis("equal")

    plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="blue")
    plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="red")

    clear_output(wait=True)
    plt.show()
    plt.plot(range(len(loss)), loss)
    plt.show()
    time.sleep(0.5)
