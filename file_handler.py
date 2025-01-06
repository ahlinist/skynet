import json
import pandas as pd


class FileHandler:
    def read_json(self, path_to_file):
        with open(path_to_file, 'r') as file:
            return json.load(file)

    def write_json(self, path_to_file, data):
        with open(path_to_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def read_csv_data(self, path_to_file):
        with open(path_to_file, mode='r') as file:
            df = pd.read_csv(file)
        return df.iloc[1:].iterrows()

    def read_csv_metadata(self, path_to_file):
        with open(path_to_file, mode='r') as file:
            df = pd.read_csv(file)

        row_count = len(df)
        min_values = df.min()
        max_values = df.max()

        header = df.columns.tolist()
        input_labels = [col for col in header if col.startswith('i')]
        output_labels = [col for col in header if col.startswith('o')]

        return row_count, input_labels, output_labels, min_values, max_values
