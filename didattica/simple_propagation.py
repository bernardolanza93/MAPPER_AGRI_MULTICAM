import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Constants
num_steps = 10   # Number of time steps
true_position = np.array([0, 0, 0])   # True initial position of the robot (x, y, theta)
velocity = 1   # Constant velocity along the x-axis
movement_std_dev = 0.01   # Standard deviation of robot movement (motion model)

sensor_std_dev_x = 0.01   # Standard deviation of the sensor measurement noise
sensor_std_dev_y = 0.01   # Standard deviation of the sensor measurement noise
sensor_std_dev_theta = 1   # Standard deviation of the sensor measurement noise

# Function to simulate robot movement with velocity
def move_robot(position):
    #simulo movimento su sbarra semi rigida (incertezza solo su theta)  ma posso anche metterne dentro altra.
    x, y, theta = position
    x += velocity * np.cos(theta)
    y += velocity * np.sin(theta)
    theta += np.random.normal(0, movement_std_dev)
    return np.array([x, y, theta])

# Function to simulate sensor measurements with noise
def sensor_measurement(position):
    x, y, theta = position
    return np.array([x + np.random.normal(0, sensor_std_dev_x),
                     y + np.random.normal(0, sensor_std_dev_y),
                     theta + np.random.normal(0, sensor_std_dev_theta)])

# Simulation
true_positions = [true_position]
estimated_positions = [true_position]

# List to store measurements for covariance calculation at each time step
measurements = [sensor_measurement(true_position)]

# List to store covariance matrices for each time step
cov_matrices_measurements = []

for _ in range(num_steps):
    true_position = move_robot(true_position)
    true_positions.append(true_position)

    # Simulate sensor measurements with noise
    measured_position = sensor_measurement(true_position)
    estimated_positions.append(measured_position)
    measurements.append(measured_position)

    # Calculate the covariance matrix for the measurements at this time step
    cov_matrix_measurements = np.cov(np.array(measurements), rowvar=False)
    cov_matrices_measurements.append(cov_matrix_measurements)

# Convert lists to arrays for easy indexing
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Calculate the difference of each measurement with respect to the true position for each variable (x, y, theta) over time
diff_x = estimated_positions[:, 0] - true_positions[:, 0]
diff_y = estimated_positions[:, 1] - true_positions[:, 1]
diff_theta = estimated_positions[:, 2] - true_positions[:, 2]

# Plot the difference of each variable over time
time_steps = np.arange(num_steps + 1)



# Plot the true positions and sensor measurements with uncertainty ellipses
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Positions', color='blue')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], label='Sensor Measurements', color='green')

# Plot the uncertainty ellipses for each covariance matrix at each time step
for t in range(num_steps):
    cov_matrix_measurements = cov_matrices_measurements[t]
    position_estimate = measurements[t][:2]
    ellipse = Ellipse(xy=position_estimate, width=2 * np.sqrt(cov_matrix_measurements[0, 0]),
                      height=2 * np.sqrt(cov_matrix_measurements[1, 1]),
                      angle=np.degrees(np.arctan2(cov_matrix_measurements[1, 0], cov_matrix_measurements[0, 0])),
                      edgecolor='r', fc='None')
    plt.gca().add_patch(ellipse)

plt.xlabel('X Position')
plt.ylim(-1,1)
plt.ylabel('Y Position')
plt.legend()
plt.title('True Positions vs. Sensor Measurements with Uncertainty Ellipses for Measurements')
plt.grid()
plt.show()


plt.plot(time_steps, diff_x, label='Difference X', linestyle='--', color='red')
plt.plot(time_steps, diff_y, label='Difference Y', linestyle='--', color='blue')
plt.plot(time_steps, diff_theta, label='Difference Theta', linestyle='--', color='green')

plt.xlabel('Time Step')
plt.ylabel('Difference')
plt.legend()
plt.title('Difference of X, Y, and Theta Measurements from True Position over Time')
plt.grid()
plt.show()