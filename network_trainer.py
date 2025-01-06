from neural_network import NeuralNetwork


class NetworkTrainer:
    def __init__(self, data_transformer, file_handler):
        self.data_transformer = data_transformer
        self.file_handler = file_handler

    def train(self, epochs_number):
        # TODO: read params like that from config file
        activation_function_name = 'relu'
        input_file_path = "dataset.csv"
        row_count, input_labels, output_labels, min_values, max_values = self.file_handler.read_csv_metadata(input_file_path)

        layers = [16, 8, len(output_labels)]

        network = NeuralNetwork(network_inputs=len(input_labels), layers=layers, activation_function_name=activation_function_name)

        for epoch in range(epochs_number):
            mse = 0.0
            data = self.file_handler.read_csv_data(input_file_path)
            for _, row in data:
                values = row.values
                mse += network.propagate_back(
                    self.__normalize_data(values, 0, input_labels, min_values, max_values),
                    self.__normalize_data(values, len(input_labels), output_labels, min_values, max_values)
                ) / row_count
            if epoch % 1 == 0:
                print('Epoch #' + str(epoch) + ' MSE=' + str(mse))

        network.print_weights()

        data = {
            "metadata": {
                "input_labels": input_labels, "output_labels": output_labels,"layers": layers,
                "activation_function": activation_function_name, "min_values": min_values.to_dict(),
                "max_values": max_values.to_dict()}, "weights": self.__build_weight_matrix(network.network),
        }

        self.file_handler.write_json('network.json', data)

    def __build_weight_matrix(self, network):
        return [[perceptron.weights.tolist() for perceptron in layer] for layer in network]

    def __normalize_data(self, row, start_from_index, labels, mins, maxes):
        return [
            self.data_transformer.normalize(float(row[i]), mins[label], maxes[label])
            for i, label in enumerate(labels, start_from_index)
        ]
