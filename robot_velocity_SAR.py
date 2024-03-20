import pandas as pd
import matplotlib.pyplot as plt
import sys


import numpy as np
from scipy.stats import linregress


def smart_cutter_df(df, threshold):


    start_idx = 0
    sub_dataframes = []
    for i in range(1, len(df)):
        if df.index[i] - df.index[i - 1] > threshold:
            # Se c'è una discontinuità
            sub_dataframes.append(df.iloc[start_idx:i])
            start_idx = i
    sub_dataframes.append(df.iloc[start_idx:])
    return sub_dataframes




def graphic_and_speed(df):


        min_timestamp = df['__time'].min()


        # Converti il timestamp in secondi sottraendo il minimo timestamp e dividendo per 1 secondo
        timestamps_in_seconds = (df['__time'] - min_timestamp) / 1.0  # 1 secondo
        x = timestamps_in_seconds
        y = df['/tf/base/tool0_controller/translation/x'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2




        # Plot dei dati di posizione come scatter plot
        plt.scatter(x, y, label='Position Data')

        # Plot della linea di regressione
        plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')

        # Aggiungi titoli e legenda
        plt.title(f'Velocità:{slope:.2f} m/s, R^2:{r_squared:.7f}')
        plt.xlabel('Timestamp')
        plt.ylabel('Traslazione X')
        plt.legend()
        plt.show()



# Carica i dati dal file CSV
data = pd.read_csv("data/1b.csv", delimiter=",")


# Estrarre il timestamp e la traslazione lungo l'asse X

min_timestamp = data['__time'].min()
# Estrai i dati filtrati
timestamps = data['__time']

# Converti il timestamp in secondi sottraendo il minimo timestamp e dividendo per 1 secondo
timestamps_in_seconds = (data['__time'] - min_timestamp) / 1.0  # 1 secondo




# Calcola il valore massimo e minimo di traslazione lungo l'asse X
x_max = data['/tf/base/tool0_controller/translation/x'].max()
x_min = data['/tf/base/tool0_controller/translation/x'].min()

# Calcola l'intervallo di X
x_range = x_max - x_min

# Imposta la soglia percentuale per eliminare i valori estremi
threshold_percent = 25  # Percentuale

# Calcola i valori soglia
threshold_value = threshold_percent / 100 * x_range

# Elimina le righe che cadono al di fuori della soglia percentuale
filtered_data = data[(data['/tf/base/tool0_controller/translation/x'] >= x_min + threshold_value) &
                     (data['/tf/base/tool0_controller/translation/x'] <= x_max - threshold_value)]





# Estrai il timestamp più piccolo per convertire il timestamp in secondi
min_timestamp = filtered_data['__time'].min()
# Estrai i dati filtrati
timestamps = filtered_data['__time']
x_translation = filtered_data['/tf/base/tool0_controller/translation/x']

# Converti il timestamp in secondi sottraendo il minimo timestamp e dividendo per 1 secondo
timestamps_in_seconds = (filtered_data['__time'] - min_timestamp) / 1.0  # 1 secondo


x_translation = filtered_data['/tf/base/tool0_controller/translation/x']


all_blocks = smart_cutter_df(filtered_data,20)
print(len(all_blocks))
for i in range(len(all_blocks)):
    print("dataset:",i)
    graphic_and_speed(all_blocks[i])
