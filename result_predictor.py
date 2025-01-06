from neural_network import NeuralNetwork


class ResultPredictor:
    def __init__(self, data_transformer, file_handler):
        self.data_transformer = data_transformer
        self.file_handler = file_handler

    def predict(self, arguments):
        data = self.file_handler.read_json('network.json')

        metadata = data['metadata']
        input_labels = metadata['input_labels']
        output_labels = metadata['output_labels']

        if len(arguments) != len(input_labels):
            #TODO: specify in the message how many params are in the dataset and in the args
            raise Exception("Number of input arguments doesn't correspond the number of model inputs")

        arg_dict = {label: arguments[index] for index, label in enumerate(input_labels)}

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
            value = float(arg_dict[label])
            min = min_values[label]
            max = max_values[label]
            normalized_inputs.append(
                self.data_transformer.normalize(value, min, max)
            )

        print('Result:')
        outputs = network.run(normalized_inputs)

        for i, label in enumerate(output_labels):
            value = outputs[i]
            min = min_values[label]
            max = max_values[label]
            print(self.data_transformer.denormalize(value, min, max))
