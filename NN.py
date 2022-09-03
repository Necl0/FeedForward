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

for i in range(1, len(layer_inputs)-1):
  for node in layer_inputs[i]:
    weight = uniform(0, 1)
    layer_weights.append([uniform(0, 1) for node in layer_inputs[i+1]])




bias = [uniform(0.0, 0.1) for j in range(layer_n)] # generate random bias values for each layer in the network
threshold = 0.5 # default treshold value, halfway from 0-1
layer_one = [*map(float, input(f"\nEnter in the inputs for the first layer ({len(layer_inputs[0])} inputs): ").split())]
layer_inputs[0] = layer_one # set inputs to first layer of network

@dataclass
class NeuralNetwork:
  """Neural Network class"""
  layer_n: int
  layer_inputs: List[List[int]]
  layer_weights: List[List[float]]
  bias: List[float]
  threshold: float

def relu( num: int) -> int:
  """Relu activation function: all layers except last"""
  return max(0, num)

def sigmoid(num: int) -> float:
  """Sigmoid activation function: last layer"""
  return 1/(1+np.exp(-num))

@dataclass
class Perceptron:
  """Perceptron Class"""
  inputs: List[int]
  weights: List[int]
  bias: List[float]
  threshold: int


def percepOutput(inputs: List[List[int]], weights: List[List[float]], bias: List[float]) -> float:
    return float(np.dot(inputs, weights) + bias)

n1 = NeuralNetwork(layer_n, layer_inputs, layer_weights, bias, threshold)

def forwardFeed(network) -> List[List[int]]:
  """Forward Feed function"""
  # iterate through each layer
  for i in range(1, network.layer_n): 
    # iterate through each perceptron  
    c = 0
    for percep in network.layer_inputs[i]:
      network.layer_inputs[i][c] = percepOutput(network.layer_inputs[i-1], network.layer_weights[i-1], network.bias[i-1])
      print(network.layer_weights[i-1])
      c+=1
  return network.layer_inputs
