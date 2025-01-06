import csv
import json
import pandas as pd
from neural_network import NeuralNetwork


class NetworkTrainer:
    def __init__(self, data_transformer):
        self.data_transformer = data_transformer

    def train(self, epochs_number):
        #TODO: to move file operations to a separate class
        with open('dataset.csv', mode='r') as file:
            df = pd.read_csv(file)

        row_count = len(df)
        min_values = df.min()
        max_values = df.max()

        header = df.columns.tolist()
        input_labels = [col for col in header if col.startswith('i')]
        output_labels = [col for col in header if col.startswith('o')]

        layers = [16, 8, len(output_labels)]

        #TODO: to read params like that from config file
        activation_function_name = 'relu'
        network = NeuralNetwork(network_inputs=len(input_labels), layers=layers, activation_function_name=activation_function_name)

        for epoch in range(epochs_number):
            mse = 0.0
            with open('dataset.csv', mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip the header row
                row_number = 0
                for row in reader:
                    row_number += 1
                    mse += network.propagate_back(
                        self.normalize_data(row, 0, input_labels, min_values, max_values),
                        self.normalize_data(row, len(input_labels), output_labels, min_values, max_values)
                    ) / row_count
            if epoch % 1 == 0:
                print('Epoch #' + str(epoch) + ' MSE=' + str(mse))

        network.print_weights()

        data = {
            "metadata": {
                "input_labels": input_labels, "output_labels": output_labels,"layers": layers,
                "activation_function": activation_function_name, "min_values": min_values.to_dict(),
                "max_values": max_values.to_dict()}, "weights": self.build_weight_matrix(network.network),
        }

        # Write the dictionary to a JSON file
        with open('network.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def build_weight_matrix(self, network):
        return [[perceptron.weights.tolist() for perceptron in layer] for layer in network]

    def normalize_data(self, row, start_from_index, labels, mins, maxes):
        return [
            self.data_transformer.normalize(float(row[i]), mins[label], maxes[label])
            for i, label in enumerate(labels, start_from_index)
        ]
