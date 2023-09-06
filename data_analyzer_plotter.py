import matplotlib.pyplot as plt
import csv
import numpy as np
from configuration_path import *




def read_from_csv(file_path):
    volumes = []
    distances = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:  # Ensure there are at least two columns in the row
                volume = float(row[0])
                distance = float(row[1])
                volumes.append(volume)
                distances.append(distance)

    return volumes, distances


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
volumes,distances = read_from_csv(csv_file_path)

# Plot histogram from the values read from the CSV file
plot_histogram(volumes)

# Create a scatter plot
plt.scatter(distances, volumes, label='Volume vs. Distance', color='blue', marker='o')

# Add labels and a title
plt.xlabel('Distances')
plt.ylabel('Volumes')
plt.title('Volume vs. Distance')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()