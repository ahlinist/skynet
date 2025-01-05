import numpy as np


class Perceptron:
    """
    The neuron itself.
    Accepts activation_function, bias value and np.array of weights as parameters.
    """
    def __init__(self, activation_function, bias, weights):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def run(self, x):
        input_values_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.activation_function.map(input_values_sum)

    def set_weights(self, weights):
        self.weights = np.array(weights, dtype=float)


class NeuralNetwork:
    """
    A network which consists of neurons.
    @layers is a matrix of integers which represents the network structure.
    @network is a matrix of perceptrons
    """
    def __init__(self, network_inputs, layers, activation_function_name='relu', init_function_name='random', eta=0.01, bias=1.0):
        self.layers = layers
        self.network = []
        self.values = []
        self.eta = eta
        self.bias = bias
        self.d = []
        self.activation_function = self.__build_activation_function(activation_function_name)

        for layer in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.d.append([])
            self.values[layer] = np.zeros(self.layers[layer], dtype=float)
            self.d[layer] = np.zeros(self.layers[layer], dtype=float)

            for neuron in range(self.layers[layer]):
                if layer == 0:
                    inputs_number = network_inputs
                else:
                    inputs_number = self.layers[layer - 1]
                self.network[layer].append(
                    Perceptron(
                        bias=self.bias,
                        activation_function=self.activation_function,
                        weights=self.__build_initial_weights(init_function_name, inputs_number)
                    )
                )

    def __build_activation_function(self, activation_function_name):
        match activation_function_name:
            case 'linear':
                return NeuralNetwork.LinearActivationFunction()
            case 'sigmoid':
                return NeuralNetwork.SigmoidActivationFunction()
            case 'relu':
                return NeuralNetwork.ReLUActivationFunction()
            case 'tanh':
                return NeuralNetwork.TanhActivationFunction()

    def __build_initial_weights(self, init_function, inputs_number):
        match init_function:
            case 'random':
                return (np.random.rand(inputs_number + 1) * 2) - 1
            case 'xavier':
                limit = np.sqrt(6 / (inputs_number + 1))
                return np.random.uniform(-limit, limit, inputs_number + 1)
            case 'he':
                stddev = np.sqrt(2 / inputs_number + 1)
                return np.random.randn(inputs_number + 1) * stddev

    def set_weights(self, weights):
        for layer in range(len(self.layers)):
            for neuron in range(self.layers[layer]):
                self.network[layer][neuron].set_weights(weights[layer][neuron])

    def print_weights(self):
        print()
        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                print('Layer: ', i + 1, ", Neuron: ", j + 1, ", Weight: ", self.network[i][j].weights)
        print()

    def run(self, x):
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                if i == 0:
                    self.values[i][j] = self.network[i][j].run(x)
                else:
                    self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        return self.values[-1]

    def propagate_back(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        output = self.run(x)
        error = y - output
        mse = sum(error ** 2) / self.layers[-1]

        self.d[-1] = self.activation_function.derivative(output) * error

        for layer_index in reversed(range(len(self.network) - 1)):
            for neuron_index in range(len(self.network[layer_index])):
                neuron_output = self.values[layer_index][neuron_index]
                downstream_gradient = sum(
                    self.network[layer_index + 1][nxt_layer_neuron_index].weights[neuron_index] *
                    self.d[layer_index + 1][nxt_layer_neuron_index]
                    for nxt_layer_neuron_index in range(len(self.network[layer_index + 1]))
                )
                self.d[layer_index][neuron_index] = (
                        self.activation_function.derivative(neuron_output) * downstream_gradient)

        for layer_index in range(len(self.network)):
            for neuron_index in range(len(self.network[layer_index])):
                input_vector = x if (layer_index == 0) else self.values[layer_index - 1]
                correction = self.eta * self.d[layer_index][neuron_index] * np.append(input_vector, self.bias)
                self.network[layer_index][neuron_index].weights += correction
        return mse

    class ActivationFunction:
        def map(self, x):
            pass

        def derivative(self, x):
            pass

    class LinearActivationFunction(ActivationFunction):
        def map(self, x):
            return x

        def derivative(self, x):
            return 1

    class SigmoidActivationFunction(ActivationFunction):
        def map(self, x):
            return 1 / (1 + np.exp(-x))

        def derivative(self, x):
            return x * (1 - x)

    class ReLUActivationFunction(ActivationFunction):
        def map(self, x):
            return np.maximum(0, x)

        def derivative(self, x):
            return np.where(x > 0, 1, 0)

    class TanhActivationFunction(ActivationFunction):
        def map(self, x):
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

        def derivative(self, x):
            return 1 - ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))) ** 2
