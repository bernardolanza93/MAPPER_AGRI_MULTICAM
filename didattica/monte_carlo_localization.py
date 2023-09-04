import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Number of particles for Monte Carlo Localization
num_particles = 1000

# Define the robot's true initial pose (x, y, theta)
true_pose = np.array([0.0, 0.0, 0.0])

# Define the motion parameters
linear_velocity = 1.0  # The robot's linear velocity (m/s)
time_interval = 0.1  # Time interval between updates (seconds)
motion_noise = 0.1  # Noise added to the robot's motion


# Simulate the robot's linear motion based on the motion parameters
def simulate_motion(pose):
    x, y, theta = pose
    x += linear_velocity * np.cos(theta) * time_interval
    y += linear_velocity * np.sin(theta) * time_interval
    theta = np.random.normal(theta, motion_noise)
    return np.array([x, y, theta])


# Simulate incremental encoder measurements with noise
def simulate_encoder_measurement_incremental(true_pose, prev_pose):
    encoder_increments = true_pose - prev_pose
    encoder_increments += np.random.normal(0, 0.1, 3)
    return encoder_increments


# Initialize particles randomly around the true pose
particles = np.random.normal(true_pose, [0.5, 0.5, 0.1], size=(num_particles, 3))

# Lists to store pose estimates and true poses for plotting
pose_estimates = [particles.mean(axis=0)]
true_poses = [true_pose]

# Perform Monte Carlo Localization for 50 time steps
num_time_steps = 50
# Lists to store covariance matrices for each time step
cov_matrices = []

# Perform Monte Carlo Localization for 50 time steps
for t in range(num_time_steps):
    # Simulate motion for each particle
    particles = np.array([simulate_motion(p) for p in particles])

    # Simulate incremental encoder measurements for each particle
    encoder_increments = np.array([simulate_encoder_measurement_incremental(p, true_poses[-1]) for p in particles])

    # Update particle weights based on the similarity to encoder increments
    weights = np.exp(-0.5 * np.sum((encoder_increments - true_poses[-1]) ** 2, axis=1))
    weights /= np.sum(weights)

    # Resampling - randomly sample particles with replacement based on weights
    indices = np.random.choice(num_particles, num_particles, p=weights)
    particles = particles[indices]

    # Store the mean pose estimate and the true pose for plotting
    pose_estimates.append(particles.mean(axis=0))
    true_pose = simulate_motion(true_pose)
    true_poses.append(true_pose)

    # Calculate covariance matrix for particles
    cov_matrix = np.cov(particles[:, :2], rowvar=False)
    cov_matrices.append(cov_matrix)

# Convert pose_estimates and true_poses lists to numpy arrays
pose_estimates = np.array(pose_estimates)
true_poses = np.array(true_poses)

# Plot the results with covariance ellipses
fig, ax = plt.subplots()
ax.plot(pose_estimates[:, 0], pose_estimates[:, 1], label="Estimated Pose")
ax.plot(true_poses[:, 0], true_poses[:, 1], label="True Pose", linestyle="--")

for t in range(num_time_steps):
    cov_matrix = cov_matrices[t]
    pose_estimate = pose_estimates[t, :2]
    ellipse = Ellipse(xy=pose_estimate, width=2 * np.sqrt(cov_matrix[0, 0]), height=2 * np.sqrt(cov_matrix[1, 1]),
                      angle=np.degrees(np.arctan2(cov_matrix[1, 0], cov_matrix[0, 0])), edgecolor='r', fc='None')
    ax.add_patch(ellipse)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Robot Localization with Incremental Encoder Measurements and Covariance Ellipses")
plt.legend()
plt.grid()
plt.axis('equal')  # Equal aspect ratio for x and y axes
plt.show()