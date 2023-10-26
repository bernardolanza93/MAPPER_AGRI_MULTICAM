import numpy as np
import matplotlib.pyplot as plt

# Define the state variables (x, y, theta) and their uncertainties
state = np.array([0.0, 0.0, 0.0])  # Initial state: (x, y, theta)
true_state = np.array([0.0, 0.0, 0.0])  # Initial state: (x, y, theta)
state_covariance = np.diag([0.0, 0.0, 0.0])  # Initial uncertainty: (x, y, theta)

# Define the motion model (how the state evolves over time)
def predict_state(state, delta_t, velocity, angular_velocity):
    x, y, theta = state
    x += delta_t * velocity * np.cos(theta)
    y += delta_t * velocity * np.sin(theta)
    theta += delta_t * angular_velocity
    return np.array([x, y, theta])

# Define the measurement model (how sensor measurements relate to the state)
def measurement_model(state):
    return state


# Create lists to store the data for plotting
true_trajectory = []
measured_positions = []
estimated_positions = []
uncertainty_x = []
uncertainty_y = []
uncertainty_theta = []


# Define the measurement uncertainty (sensor noise)
measurement_covariance = np.diag([0.46, 0.35, 0.03])

true_covariance =  np.diag([0.01, 0.01, 0.001])

# Simulate the noisy circular path motion
num_steps = 200

# Simulate the noisy circular path motion and Kalman Filter updates
for i in range(num_steps):
    # Simulate motion
    delta_t = 0.2  # Time step
    if i < int(num_steps/2):
        angular_velocity = 0.05  # Angular velocity
        velocity = 1.0  # Linear velocity

    else:
        angular_velocity = -0.1  # Angular velocity
        velocity = 3.0  # Linear velocity

    true_state = predict_state(true_state, delta_t, velocity, angular_velocity) #+ np.random.multivariate_normal([0, 0, 0], true_covariance)
    true_trajectory.append(true_state[0:2])

    # Simulate noisy measurements
    noisy_measurement = true_state + np.random.multivariate_normal([0, 0, 0], measurement_covariance)

    # Kalman Filter Prediction Step
    state = predict_state(state, delta_t, velocity, angular_velocity)
    state_covariance = state_covariance  # Update based on the motion model (skip for simplicity)

    # Kalman Filter Update Step
    kalman_gain = state_covariance @ np.linalg.inv(state_covariance + measurement_covariance)
    state = state + kalman_gain @ (noisy_measurement - measurement_model(state))
    state_covariance = (np.eye(3) - kalman_gain) @ state_covariance

    # Store the data for plotting

    measured_positions.append(noisy_measurement[0:2])
    estimated_positions.append(state[0:2])
    uncertainty_x.append(np.sqrt(state_covariance[0, 0]))
    uncertainty_y.append(np.sqrt(state_covariance[1, 1]))
    uncertainty_theta.append(np.sqrt(state_covariance[2, 2]))

# Extract x and y coordinates for plotting
true_x, true_y = zip(*true_trajectory)
measured_x, measured_y = zip(*measured_positions)
estimated_x, estimated_y = zip(*estimated_positions)

# Plot the true trajectory, measured positions, and estimated positions
plt.figure(figsize=(10, 6))
plt.plot(true_x, true_y, label="True Trajectory", color='blue')
plt.scatter(measured_x, measured_y, label="Measured Positions", color='red', marker='x')
plt.scatter(estimated_x, estimated_y, label="Estimated Positions", color='orange', marker='o')

# Plot uncertainty evolution for x, y, and theta
plt.figure(figsize=(10, 4))
plt.plot(uncertainty_x, label="Uncertainty X", color='blue')
plt.plot(uncertainty_y, label="Uncertainty Y", color='red')
plt.plot(uncertainty_theta, label="Uncertainty Theta", color='green')
plt.xlabel("Time Step")
plt.ylabel("Uncertainty")
plt.legend()

# Set axis labels and legend
plt.figure(1)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.title("Robot Trajectory with Measured Positions")
plt.grid()

# Display the plots
plt.show()
