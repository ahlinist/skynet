import json
import argparse
from neural_network import NeuralNetwork


def main():
    with open('network.json', 'r') as file:
        data = json.load(file)

    metadata = data['metadata']
    input_labels = metadata['input_labels']
    output_labels = metadata['output_labels']

    parser = argparse.ArgumentParser(description='Process two decimal values')
    for label in input_labels:
        parser.add_argument(label, type=float, help='coefficient ' + label)

    cli_args = parser.parse_args()

    min_values = metadata['min_values']
    max_values = metadata['max_values']
    layers = metadata['layers']
    activation_function = metadata['activation_function']
    weights = data['weights']

    network = NeuralNetwork(inputs=len(input_labels), layers=layers, activation_function=activation_function)

    for i, layer in enumerate(network.network):
        for j, perceptron in enumerate(layer):
            perceptron.weights = weights[i][j]

    normalized_inputs = []

    for input_label in input_labels:
        normalized_inputs.append(
            normalize(getattr(cli_args, input_label), input_label, min_values, max_values)
        )

    print('Result:')
    roots = network.run(normalized_inputs)

    for i, output_label in enumerate(output_labels):
        print(denormalize(roots[i], output_label, min_values, max_values))

def normalize(value, label, mins, maxes):
    min_value = mins[label]
    max_value = maxes[label]
    return (value - min_value)/(max_value - min_value)

def denormalize(value, label, mins, maxes):
    min_value = mins[label]
    max_value = maxes[label]
    return (value * (max_value - min_value)) + min_value

if __name__ == '__main__':
    main()
