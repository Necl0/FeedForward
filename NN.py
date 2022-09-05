from dataclasses import dataclass
from typing import List
import numpy as np
from random import uniform

layer_n = int(input("How many layers are in the network? "))
layer_inputs, layer_weights = [], []

# get the number of perceptrons/nodes per hidden layer
for layer in range(1, layer_n+1):
  nodes = int(input(f"How many nodes in layer {layer}? "))
  layer_inputs.append([float() for i in range(nodes)])



bias = [uniform(0.0, 0.1) for j in range(layer_n)] # generate random bias values for each layer in the network
threshold = 0.5 # default treshold value, halfway from 0-1
layer_one = [*map(float, input(f"\nEnter in the inputs for the first layer ({len(layer_inputs[0])} inputs): ").split())]
layer_inputs[0] = layer_one # set inputs to first layer of network

for i in range(2, layer_n+1):
  for node in layer_inputs[i-1]:
    layer_weights.append([uniform(0,1) for i in range(len(layer_inputs[i-2]))])
    

@dataclass
class NeuralNetwork:
  """Neural Network dataclass"""
  layer_n: int
  layer_inputs: List[List[int]]
  layer_weights: List[List[float]]
  bias: List[float]
  threshold: float

def relu( num: int) -> int:
  """Relu activation: all layers except last"""
  return max(0, num)

def sigmoid(num: int) -> float:
  """Sigmoid activation: last layer"""
  return 1/(1+np.exp(-num))

@dataclass
class Perceptron:
  """Perceptron Class"""
  inputs: List[int]
  weights: List[int]
  bias: List[float]
  threshold: int


def percepOutput(inputs: List[List[int]], weights: List[List[float]], bias: List[float]) -> float:
    """Perceptron output"""
    return float(np.dot(inputs, weights) + bias)

n1 = NeuralNetwork(layer_n, layer_inputs, layer_weights, bias, threshold)

def forwardFeed(network) -> List[List[int]]:
  """Forward Feed"""
  c = 0
  for n in range(1, network.layer_n):
    a = 0
    for _ in network.layer_inputs[n]:
      network.layer_inputs[n][a] = percepOutput(network.layer_inputs[n-1], layer_weights[c] , network.bias[n-1])
      a+=1
      c+=1
  return network.layer_inputs[-1]

output = forwardFeed(n1)

print(f"The output from the Neural Network is {output}")
