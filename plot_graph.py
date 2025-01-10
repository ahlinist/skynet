import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from result_predictor import ResultPredictor
from data_transformer import DataTransformer
from file_handler import FileHandler

data_transformer = DataTransformer()
file_handler = FileHandler()
result_predictor = ResultPredictor(data_transformer=data_transformer, file_handler=file_handler)

with open('dataset.csv', mode='r') as file:
    dfx = pd.read_csv(file, skipinitialspace=True, usecols=['o1'])

with open('dataset.csv', mode='r') as file:
    dfy = pd.read_csv(file, skipinitialspace=True, usecols=['o2'])

with open('dataset_aux.csv', mode='r') as file:
    dfx_aux = pd.read_csv(file, skipinitialspace=True, usecols=['o1'])

with open('dataset_aux.csv', mode='r') as file:
    dfy_aux = pd.read_csv(file, skipinitialspace=True, usecols=['o2'])

true_x = dfx.to_numpy()
true_y = dfy.to_numpy()
plt.scatter(true_x, true_y, c='blue')

aux_x = dfx_aux.to_numpy()
aux_y = dfy_aux.to_numpy()
plt.scatter(aux_x, aux_y, c='green')

inferred_x = []
inferred_y = []

for i in range(-10, 30):
    result = result_predictor.predict([1000000000000 / 2 * i])
    inferred_x.append(result[0])
    inferred_y.append(result[1])

plt.scatter(np.array(inferred_x), np.array(inferred_y), c='red')

plt.show()
