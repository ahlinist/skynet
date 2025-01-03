import random
import math
import csv

DATASET_SIZE = 10000
MIN_TIME = 0 #s
MAX_TIME = 1000 #s

G = 6.67430e-11  # gravitational constant m^3 kg^-1 s^-2
CENTRAL_BODY_MASS = 100000000000 #kg
START_X_VELOCITY = 0.0 #m/s
START_Y_VELOCITY = 1.155 #m/s
START_X_POSITION = 5
START_Y_POSITION = 0

def main():
    with open(r'dataset.csv', 'w', newline='') as csvfile:
        csvfile.truncate()
        fieldnames = ['i1', 'o1', 'o2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'i1': 'i1', 'o1': 'o1', 'o2': 'o2'})

    velocity = math.sqrt(G * CENTRAL_BODY_MASS / START_X_POSITION)
    radius = math.sqrt(START_X_POSITION ** 2 + START_Y_POSITION ** 2)
    circumference = 2 * math.pi * radius
    period = circumference / velocity

    for i in range(DATASET_SIZE):
        time = random.uniform(MIN_TIME, MAX_TIME)
        angle = 2 * math.pi * time / period
        x = math.cos(angle) * radius
        y = math.sin(angle) * radius

        with open(r'dataset.csv', 'a', newline='') as csvfile:
            fieldnames = ['i1', 'o1', 'o2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'i1': time, 'o1': x, 'o2': y})

if __name__ == '__main__':
    main()
