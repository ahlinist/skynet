import json
import argparse
from neural_network import NeuralNetwork
from data_transformer import DataTransformer


def main():
    data_transformer = DataTransformer()
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
    activation_function_name = metadata['activation_function']
    weights = data['weights']

    network = NeuralNetwork(network_inputs=len(input_labels), layers=layers, activation_function_name=activation_function_name)

    for i, layer in enumerate(network.network):
        for j, perceptron in enumerate(layer):
            perceptron.weights = weights[i][j]

    normalized_inputs = []

    for label in input_labels:
        value = getattr(cli_args, label)
        min = min_values[label]
        max = max_values[label]
        normalized_inputs.append(
            data_transformer.normalize(value, min, max)
        )

    print('Result:')
    outputs = network.run(normalized_inputs)

    for i, label in enumerate(output_labels):
        value = outputs[i]
        min = min_values[label]
        max = max_values[label]
        print(data_transformer.denormalize(value, min, max))

if __name__ == '__main__':
    main()
