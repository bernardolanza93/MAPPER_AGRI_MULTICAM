import sys

import matplotlib.pyplot as plt
import csv
import numpy as np
from CONFIGURATION_VISION import *
import math
import seaborn as sns
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#file_to_read = csv_file_path
REAL_VOLUME =  15.74 * 1000 / 0.72
print(REAL_VOLUME)
file_to_read = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/data/run6_iter4/data_volume_iteration_5.csv"



def z_score_norm(z_values, threshold_factor = 0.6):
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)

    # Calculate z-scores
    z_scores = (z_values - z_mean) / z_std

    # Apply a correction to the z-values based on their z-scores
    corrected_z_values = z_mean + (z_scores * threshold_factor * z_std)
    return corrected_z_values


def clean_lists_based_on_volume_outliers(volumes, distances, threshold=1.7):
    # Calculate the mean and standard deviation of the volumes
    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)

    # Identify and remove outliers beyond the specified threshold
    cleaned_volumes = []
    cleaned_distances = []
    for i, volume in enumerate(volumes):
        if abs(volume - mean_volume) <= threshold * std_volume:
            cleaned_volumes.append(volume)
            cleaned_distances.append(distances[i])

    # Calculate the standard deviation of the cleaned volumes
    cleaned_std_volume = np.std(cleaned_volumes)

    return cleaned_volumes, cleaned_distances

def calculate_mean_deviation(measurements, true_value):
    # Convert the measurements to a NumPy array for ease of computation
    measurements = np.array(measurements)

    # Calculate the absolute deviations between measurements and true_value
    absolute_deviations = np.abs(measurements - true_value)

    # Calculate the mean of absolute deviations
    mean_deviation = np.mean(absolute_deviations)

    return mean_deviation
def calculate_rmse(measurements, true_value):

    # Convert the measurements to a NumPy array for ease of computation
    measurements = np.array(measurements)

    # Calculate the squared differences between measurements and true_value
    squared_errors = (measurements - true_value) ** 2

    # Calculate the mean of squared errors
    mean_squared_error = np.mean(squared_errors)

    # Calculate the square root of the mean squared error to get RMSE
    rmse = np.sqrt(mean_squared_error)

    return rmse

def density_determination():
    # Given volume and mass lists
    volume_or = [1308.649269,2677.854308,1463.196778,2997.275741,628.9060085,453.0726385,1931.101661,1691.450763,646.9167592,1323.578118,1220.306113,1475.205516,853.1026267,910.433551]
    volume =  [x / 1000 for x in volume_or]
    mass = [0.96,1.86,1.21,2.19,0.45,0.32,1.31,1.07,0.45,0.85,0.83,1.04,0.55,0.52]


    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(volume, mass)

    # Calculate the regression line
    regression_line = [(slope * x) + intercept for x in volume]

    # Calculate the uncertainty in mass and volume
    std_mass = std_err * np.sqrt(len(mass))
    std_volume = std_err * np.sqrt(np.sum(np.square(np.array(volume) - np.mean(volume))))

    # Calculate density
    density = np.array(mass) / np.array(volume)

    # Calculate uncertainty in density estimation
    uncertainty_density = density * np.sqrt((std_mass / np.array(mass)) ** 2 + (std_volume / np.array(volume)) ** 2)

    # Calculate the mean density and its uncertainty
    mean_density = np.mean(density)
    uncertainty_mean_density = np.std(density) / np.sqrt(len(density))

    # Plot the data points and regression line
    plt.scatter(volume, mass, label='Data Points')
    plt.plot(volume, regression_line, color='red', label=f'Regression Line (y={slope:.2f}x + {intercept:.2f})')
    plt.xlabel('Volume (mm³)')
    plt.ylabel('Mass (g)')

    # Include mean density and its uncertainty in the title
    plt.title(f'Volume vs. Mass\nMean Density: {mean_density:.2f} ± {uncertainty_mean_density:.2f} g/cm³')

    plt.legend()
    plt.grid(True)
    plt.show()



def plot_scatter_with_horizontal_line(distance_mm, volume, horizontal_line_y, horizontal_line_label):


    volume = [x / 1000 for x in volume]

    # Create a scatter plot using Seaborn
    sns.scatterplot(x=distance_mm, y=volume, label='Observations')


    # Given values and uncertainties
    mass = 15.74  # g
    density = 0.72  # g/cm³
    mass_uncertainty = 0.01  # g
    density_uncertainty = 0.04  # g/cm³

    # Calculate volume
    volume_r = ( mass / density )
    #print("vol",volume)

    # Propagate uncertainties
    relative_mass_uncertainty = mass_uncertainty / mass
    relative_density_uncertainty = density_uncertainty / density

    uncertainty_volume_mass = relative_mass_uncertainty * volume_r
    uncertainty_volume_density = relative_density_uncertainty * volume_r

    # Total uncertainty using RSS method
    total_uncertainty_volume = math.sqrt(uncertainty_volume_mass ** 2 + uncertainty_volume_density ** 2)

    # Calculate the mean and standard deviation of the volume
    mean_volume = np.mean(volume)
    std_volume = np.std(volume)
    plt.axhline(y=mean_volume, color='red', linestyle='--', label="mean")

    # Add a horizontal line
    error = total_uncertainty_volume
    plt.axhline(y=volume_r, color='blue', linestyle='--', label="ground truth")
    plt.errorbar(0, volume_r, xerr=0, yerr=error, color='blue', linestyle='-', marker='o', markersize=5, capsize=5)
    plt.axhspan(volume_r + total_uncertainty_volume, volume_r - total_uncertainty_volume, color='lightblue', alpha=0.5, label='uncertainty')
    plt.axhline(y=volume_r + total_uncertainty_volume, color='blue', linestyle='--',linewidth=0.5)
    plt.axhline(y=volume_r - total_uncertainty_volume, color='blue', linestyle='--',linewidth=0.5)


    #plt.axhspan(mean_volume + std_volume, mean_volume - std_volume, color='lightblue', alpha=0.5, label='uncertainty')





    rmse = calculate_rmse(volume,volume_r)
    rmse_pc = (rmse/volume_r )*100
    mean_dev = calculate_mean_deviation(volume,volume_r)
    mean_dev_pc = (mean_dev / volume_r) * 100

    print("rmse",rmse,"rmse_pc",rmse_pc ,"mean_dev",
    mean_dev ,"mean_dev_pc",
    mean_dev_pc)

    # Define the title with mean and standard deviation
    title = f'Mean Volume: {mean_volume:.1f} cm$^3$, \n Std-Dev: {std_volume:.1f} cm$^3$'

    # Set y-axis limits
    plt.ylim(10, 45)
    plt.xlim(500,1200)

    # Add labels and title
    plt.xlabel('Distance [mm]', fontsize=14)
    plt.ylabel('Volume [cm$^3$]', fontsize=14)
    plt.title(title, fontsize=14)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def volume(L, d):
    if L is not None and d is not None:
        vol = (math.pi * pow(d, 2) * L) / 4
        return vol
    else:
        return 0

def advanced_draw_histogram(data):
    data = [x / 1000 for x in data]
    # Create a histogram using Seaborn
    sns.histplot(data, kde=True, color='blue', bins=20)  # Customize as needed

    # Add labels and title
    plt.xlabel('Branch total volume error [cm$^3$]', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)


    # Show the plot
    plt.show()

def read_from_csv(file_to_read):
    print(file_to_read)
    volumes = []
    distances = []

    with open(file_to_read, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:  # Ensure there are at least two columns in the row
                volume = float(row[0])
                distance = float(row[1])
                volumes.append(volume)
                distances.append(distance)

    return volumes, distances


def plot_histogram(values):
    values = [x / 1000 for x in values]
    plt.hist(values, bins=100, edgecolor='black')
    mean_vol = np.mean(values)
    plt.vlines(mean_vol, 0, 50, linestyles ="dashed", colors ="g")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values, mean:' +  str(int(mean_vol)) + " STD:" + str(int(np.std(values))) + " OBS:"+ str(len(values)))
    plt.show()


# ls = [53,87,150,88.5,57.5,14,11.5,15,13,12,11.8]
# ds = [6.5,6.8,8,7.9,8.2,7,11,12.5,12,11.5,12]
# lenght = 0
# volume_tot = 0
#
# print(len(ds),len(ls))
#
# if len(ls) == len(ds):
#     for i in range(len(ds)):
#         lenght  += ls[i]
#         vol_i = int(volume(ls[i],ds[i]))
#
#         volume_tot +=  vol_i
#
# print("vol:",volume_tot, " length:",lenght)
# sys.exit()
# #27349
# #513

# print("volume app:", volume(515,14))



#density_determination()



print("reading volumes...")
# Read values from the CSV file
volumes,distances = read_from_csv(file_to_read)
volumes = z_score_norm(volumes)
#volumes,distances = clean_lists_based_on_volume_outliers(volumes,distances)

# Plot histogram from the values read from the CSV file
scarti_vol = [x - REAL_VOLUME for x in volumes]
advanced_draw_histogram(scarti_vol)


plot_scatter_with_horizontal_line(distances,volumes,REAL_VOLUME,"Ground truth")
#plot_histogram(volumes)

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