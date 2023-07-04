import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math


def read_csv_files(folder_path):
    frame_numbers = []
    date_and_hours = []
    positions = []
    speeds = []
    orientations = []

    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            xxxx = []
            yyyy = []


            with open(file_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) == 0:
                        continue

                    frame_number = int(row[0])
                    frame_numbers.append(frame_number)

                    date_and_hour = row[1]
                    date_and_hours.append(date_and_hour)

                    position_values = row[2].split(',')
                    x_pos = float(position_values[0].split(':')[1])
                    y_pos = float(position_values[1].split(':')[1])
                    z_pos = float(position_values[2].split(':')[1])

                    positions.append((x_pos, y_pos, z_pos))
                    xxxx.append(x_pos)
                    yyyy.append(y_pos)


                    speed_values = row[3].split(',')
                    x_speed = float(speed_values[0].split(':')[1])
                    y_speed = float(speed_values[1].split(':')[1])
                    z_speed = float(speed_values[2].split(':')[1])
                    speeds.append((x_speed, y_speed, z_speed))

                    orientation_values = row[4].strip('[]').split(',')
                    x_orientation = float(orientation_values[0])
                    y_orientation = float(orientation_values[1])
                    z_orientation = float(orientation_values[2])
                    orientations.append((x_orientation, y_orientation, z_orientation))


            xxx = yyyy
            yyy = xxxx
            fig, (ax1) = plt.subplots(1, 1)
            ax1.scatter(xxx, yyy)

            ax1.grid(True)
            # obtain m (slope) and b(intercept) of linear regression line
            print(len(xxx))
            #x_pos = [float(i) for i in x_pos]

            mrr, brr = np.polyfit(xxx, yyy, 1)

            y_estimated = [(x * mrr) + brr for x in xxx]
            y_residuals = [a_i - b_i for a_i, b_i in zip(y_estimated, xxx)]
            squared_res = [i ** 2 for i in y_residuals]
            MSE = statistics.mean(squared_res)
            RMSE = math.sqrt(MSE)
            print("mse rmse", MSE, RMSE)

            mr, br = np.polyfit(xxx, yyy, 1, cov=True)
            dm = np.sqrt(br[0][0])
            db = np.sqrt(br[1][1])

            print("m: {} +/- {}".format(mr[0], np.sqrt(br[0][0])))
            print("b: {} +/- {}".format(mr[1], np.sqrt(br[1][1])))

            ax1.set_title(
                "R = m*Z + b : RMSE=" + str(round(RMSE, 4)) + " \n m=" + str(round(mrr, 7)) + "+/-" + str(
                    round(dm, 7)) + "; b=" + str(round(brr, 3)) + "+/-" + str(round(db, 3)))

            # add linear regression line to scatterplot

            # ax3.plot(Z_all,[x * m for x in Z_all]+b,color='g', linestyle='--')
            ax1.plot(xxx, [x * mrr for x in xxx] + brr, color='r', linestyle='-.')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')

            fig.suptitle("analysis of results")

            plt.show()






    return frame_numbers, date_and_hours, positions, speeds, orientations




folder_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/data/old'
frame_numbers, date_and_hours, positions, speeds, orientations = read_csv_files(folder_path)
