import csv
import json


class FileHandler:
    def read_json(self, path_to_file):
        with open(path_to_file, 'r') as file:
            return json.load(file)

    def write_json(self, path_to_file, data):
        with open(path_to_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)
