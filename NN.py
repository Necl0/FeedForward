from dataclasses import dataclass
from typing import List
import numpy as np
from random import uniform

layer_n = int(input("How many layers are in the network? "))
layer_inputs, layer_weights = [], []

for layer in range(1, layer_n+1):
  nodes = int(input(f"How many nodes in layer {layer}? "))
  layer_inputs.append([float() for i in range(nodes)])
  layer_weights.append([uniform(0, 1) for i in range((nodes))])


bias = [uniform(0.0, 0.1) for j in range(layer_n)]
threshold = float(input("Enter in the threshold: "))

layer_one = [*map(float, input(f"\nEnter in the inputs for the first layer ({len(layer_inputs[0])} inputs): ").split())]
layer_inputs[0] = layer_one

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


def percep_output(inputs, weights, bias, treshold) -> int:
    return int(relu(np.dot(inputs, weights) + bias > threshold))

n1 = NeuralNetwork(layer_n, layer_inputs, layer_weights, bias, threshold)

# iterate through each layer
for i in range(1, layer_n): 
  # iterate through each perceptron  
  c = 0
  for percep in layer_inputs[i]:
    layer_inputs[i][c] = percep_output(layer_inputs[i-1], layer_weights[i-1], bias[i-1], 0.5)
    c+=1

print(f"Output: {layer_inputs[-1][0]}")
