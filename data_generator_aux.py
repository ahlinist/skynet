import random
import math
import csv
import numpy as np

DATASET_SIZE = 1000000

MIN_TIME = 400000000000 #s
MAX_TIME = 7000000000000 #s

SEMIMINOR_AXIS = 1000000000
SEMIMAJOR_AXIS = 1030000000

ECCENTRICITY = math.sqrt(1 - SEMIMINOR_AXIS ** 2 / SEMIMAJOR_AXIS ** 2)
PERIOD = 10000000000000

def main():
    with open(r'dataset_aux.csv', 'w', newline='') as csvfile:
        csvfile.truncate()
        fieldnames = ['i1', 'o1', 'o2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'i1': 'i1', 'o1': 'o1', 'o2': 'o2'})

    for i in range(100):
        time = i * PERIOD / 100

        if time > MIN_TIME and time < MAX_TIME:
            continue

        M = calculate_mean_anomaly(time)
        E = calculate_eccentric_anomaly(M, ECCENTRICITY)
        theta = calculate_true_anomaly(E, ECCENTRICITY)

        x = SEMIMINOR_AXIS * math.cos(theta)
        y = SEMIMAJOR_AXIS * math.sin(theta)

        with open(r'dataset_aux.csv', 'a', newline='') as csvfile:
            fieldnames = ['i1', 'o1', 'o2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'i1': time, 'o1': x, 'o2': y})

def calculate_mean_anomaly(time):
    return 2 * math.pi * time / PERIOD


def calculate_eccentric_anomaly(M, e, tol=1e-6, max_iter=100):
    # Initial guess for E (can use M for small eccentricities)
    E = M if e < 0.8 else np.pi  # Better guess for higher e

    for _ in range(max_iter):
        # Compute the function value and its derivative
        f_E = E - e * np.sin(E) - M
        f_E_prime = 1 - e * np.cos(E)

        # Newton-Raphson step
        delta_E = -f_E / f_E_prime
        E += delta_E

        # Check for convergence
        if abs(delta_E) < tol:
            return E

    raise RuntimeError("Kepler's equation did not converge after maximum iterations.")


def calculate_true_anomaly(E, e):
    # Calculate true anomaly using the formula
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    # Ensure the result is in the correct range [0, 2π]
    theta = theta % (2 * np.pi)

    return theta


if __name__ == '__main__':
    main()
