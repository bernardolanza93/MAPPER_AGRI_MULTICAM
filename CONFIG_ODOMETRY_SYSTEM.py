import os
import json
DETECT_MARKER = True
# Define the folder path
FOLDER_CALIBRATION_CAMERA = "CALIBRATION_CAMERA_FILE"
IMAGE_CALIBRATION_PATH = "CALIB_IMAGES"
board_size = (9, 6)
MAX_FRAME_TEST = 30

print("TO TERMINATE THIS PROGRAMM PRESS : CTRL + c  ")
print("if the program is not correctly closed digit : pkill python, and then press RETURN")
PRINT_FPS_ODOMETRY = 1


def check_config_file(time_sleep_odometry=0.067, marker_dimension=20):
    config_file = 'config.json'
    default_values = {
        'time_sleep_odometry': time_sleep_odometry,
        'marker_dimension': marker_dimension
    }

    if not os.path.exists(config_file):
        # If the config file doesn't exist, create it with default values
        with open(config_file, 'w') as f:
            json.dump(default_values, f)
        return default_values
    else:
        # If the config file exists, read its contents
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Check and update missing or incomplete values
        updated_config = {**default_values, **config_data}
        with open(config_file, 'w') as f:
            json.dump(updated_config, f)

        return updated_config

# Example usage:
config = check_config_file(time_sleep_odometry=0.05, marker_dimension=25)  # Change default values if needed

# Accessing the values from the dictionary
DIVIDER_FPS_REDUCTION = config.get('time_sleep_odometry')
MARKER_SIZE = config.get('marker_dimension')

print(f"The value of time_sleep_odometry is: {DIVIDER_FPS_REDUCTION}")
print(f"The value of marker_dimension is: {MARKER_SIZE}")
