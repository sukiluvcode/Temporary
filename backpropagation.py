import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

weights = np.array([[0.4, 0.6], [0.3, 0.7]])
output_weights = np.array([[0.8], [0.5]])
inputs = np.array([0.5, 0.8])
hidden_bias = np.array([0.1, 0.2])
output_bias = np.array([0.3])

def feedforward(inputs, weights, hidden_bias):
    z = np.dot(inputs, weights) + hidden_bias
    return z, sigmoid(z)

hidden_neuron, output = feedforward(inputs, weights, hidden_bias)
final_neuron, final = feedforward(output, output_weights, output_bias)

derivative_final_weights = (final - 0.9) * sigmoid_derivative(final_neuron) * output
# print(derivative_final_weights)
# print(sigmoid_derivative(final_neuron) * output_weights.T * sigmoid_derivative(hidden_neuron))
print(output_weights.T)
print(sigmoid_derivative(hidden_neuron).shape)
print(output_weights.T * sigmoid_derivative(hidden_neuron))
derivative_hidden_weights = np.outer(inputs, (final - 0.9) * sigmoid_derivative(final_neuron) * output_weights.T * sigmoid_derivative(hidden_neuron))
# print(derivative_hidden_weights)