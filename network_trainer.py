import csv
import pandas as pd
from neural_network import NeuralNetwork
import json

EPOCHS_NUMBER = 3

def main():
    with open('dataset.csv', mode='r') as file:
        df = pd.read_csv(file)

    row_count = len(df)
    min_values = df.min()
    max_values = df.max()

    header = df.columns.tolist()
    input_labels = [col for col in header if col.startswith('i')]
    output_labels = [col for col in header if col.startswith('o')]

    layers = [16, 8, len(output_labels)]
    activation_function = 'relu'
    network = NeuralNetwork(inputs=len(input_labels), layers=layers, activation_function=activation_function)

    for epoch in range(EPOCHS_NUMBER):
        mse = 0.0
        with open('dataset.csv', mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row
            row_number = 0
            for row in reader:
                row_number += 1
                mse += network.propagate_back(
                    normalize_input(row, input_labels, min_values, max_values),
                    normalize_output(row, input_labels, output_labels, min_values, max_values)
                ) / row_count
        if epoch % 1 == 0:
            print('Epoch #' + str(epoch) + ' MSE=' + str(mse))

    network.print_weights()

    data = {
        "metadata": {
            "input_labels": input_labels, "output_labels": output_labels,"layers": layers,
            "activation_function": activation_function, "min_values": min_values.to_dict(),
            "max_values": max_values.to_dict()}, "weights": build_weight_matrix(network.network),
    }

    # Write the dictionary to a JSON file
    with open('network.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

def build_weight_matrix(network):
    return [[perceptron.weights.tolist() for perceptron in layer] for layer in network]

def normalize_input(row, input_labels, mins, maxes):
    result = []
    for i, label in enumerate(input_labels):
        min_value = mins[label]
        max_value = maxes[label]
        result.append((float(row[i]) - min_value) / (max_value - min_value))
    return result

def normalize_output(row, input_labels, output_labels, mins, maxes):
    result = []
    for i, label in enumerate(output_labels, len(input_labels)):
        min_value = mins[label]
        max_value = maxes[label]
        result.append((float(row[i]) - min_value) / (max_value - min_value))
    return result

if __name__ == '__main__':
    main()
