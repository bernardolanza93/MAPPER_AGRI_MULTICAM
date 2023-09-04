import matplotlib.pyplot as plt
import csv
import numpy as np
from configuration_path import *

def read_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        values = [float(row[0]) for row in reader]  # Assuming the values are numeric
    return values


def plot_histogram(values):
    plt.hist(values, bins=100, edgecolor='black')
    mean_vol = np.mean(values)
    plt.vlines(mean_vol, 0, 50, linestyles ="dashed", colors ="g")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values, mean:' +  str(int(mean_vol)) + " STD:" + str(int(np.std(values))))
    plt.show()




print("reading volumes...")
# Read values from the CSV file
values_from_csv = read_from_csv(csv_file_path)

# Plot histogram from the values read from the CSV file
plot_histogram(values_from_csv)