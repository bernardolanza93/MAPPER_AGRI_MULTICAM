import sys

import cv2
import numpy as np
import os
import pandas as pd
import csv
from additional_functions import *
from scipy.stats import linregress


# Salva il dataframe su un file Excel
output_excel = 'marker_data.xlsx'



# Definisci il percorso della cartella di calibrazione
folder_calibration = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/CALIBRATION_CAMERA_FILE"
PLOT_SINGLE_PATH = 0
# Carica i parametri di calibrazione della camera
mtx = np.load(os.path.join(folder_calibration, "camera_matrix.npy"))
dist = np.load(os.path.join(folder_calibration, "dist_coeffs.npy"))

video_name = 'GX010114.MP4'
# Definisci il percorso del video
video_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/'+video_name

# Definisci la dimensione del marker ArUco
#MARKER_SIZE = 0.0978 #cm piccolo prova 1
MARKER_SIZE = 0.1557 #cm grande prova 2



# Inizializza il detector di marker ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Cartella per i dati
folder_path = 'dati_marker_optical_flow'
os.makedirs(folder_path, exist_ok=True)

# Percorso del file CSV
csv_path = os.path.join(folder_path, 'dati_marker_optical_flow.csv')

# Apri il video
cap = cv2.VideoCapture(video_path)

# Inizializza il rilevatore di flusso ottico
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Definisci il colore per disegnare i vettori di flusso ottico
color = (0, 255, 0)

marker_ids = [7,8,9,10,11]

# Nome del file Excel
output_excel_res = 'results.xlsx'

# Header del file Excel
header = ['marker', 'z_mean', 'z_std', 'vx', 'vx_std', 'vx_3D', 'vx_3D_std']

# Crea un DataFrame vuoto con gli header
df_res = pd.DataFrame(columns=header)

# Salva il DataFrame con gli header nel file Excel
df_res.to_excel(output_excel_res, index=False)

print(f"Creato il file '{output_excel}' con gli header vuoti.")


def smart_cutter_df(df, threshold):
    start_idx = 0
    sub_dataframes = []
    for i in range(1, len(df)):
        if df['n_frame'].iloc[i] - df['n_frame'].iloc[i - 1] > threshold:
            # Se c'è una discontinuità
            sub_dataframes.append(df.iloc[start_idx:i])
            start_idx = i
    sub_dataframes.append(df.iloc[start_idx:])
    return sub_dataframes

def divide_dataframe(df, start_points, end_points):
    """
    Divide il dataframe in 5 parti diverse in base ai punti di inizio e fine forniti.

    Argomenti:
        df (pd.DataFrame): Il dataframe da dividere.
        start_points (list): Una lista di 5 numeri interi rappresentanti i punti di inizio per ciascuna serie.
        end_points (list): Una lista di 5 numeri interi rappresentanti i punti di fine per ciascuna serie.

    Ritorna:
        list: Una lista di 5 dataframe, ognuno contenente una serie tagliata.
    """

    # Assicurati che i punti di inizio e fine abbiano lunghezza corretta
    if len(start_points) != 10 or len(end_points) != 10:
        raise ValueError("I punti di inizio e fine devono essere forniti per ciascuna delle 5 serie.")

    # Inizializza una lista vuota per contenere i dataframe tagliati
    divided_dfs = []

    # Cicla attraverso i punti di inizio e fine e taglia il dataframe
    for start, end in zip(start_points, end_points):
        # Taglia il dataframe usando i punti di inizio e fine
        sliced_df = df[(df['n_frame'] >= start) & (df['n_frame'] <= end)]
        # Aggiungi il dataframe tagliato alla lista
        divided_dfs.append(sliced_df)


    return divided_dfs

def delete_static_data_manually(df, marker_riferimento, confidence_delation):
    # Calcola il valore massimo e minimo della posizione x del marker di riferimento
    x_min = df[marker_riferimento].min()
    x_max = df[marker_riferimento].max()

    # Calcola il range del valore di x tenendo conto della confidence delation
    x_range = x_max - x_min
    x_range *= (1 - confidence_delation)

    # Calcola i valori soglia
    x_threshold_min = x_min + confidence_delation * x_range
    x_threshold_max = x_max - confidence_delation * x_range

    # Filtra le righe che non soddisfano i criteri di soglia
    df_filtered = df[(df[marker_riferimento] >= x_threshold_min) & (df[marker_riferimento] <= x_threshold_max)]

    return df_filtered



def imaga_analizer_raw():
    # Elimina il file se già esiste
    if os.path.exists(output_excel):
        os.remove(output_excel)
    n_frames = 0

    z_m = 'z_3D_'
    x_im = 'x_ip_'
    x_3D = 'x_3D_'
    columns = ['n_frame']

    for i in marker_ids:
        columns.extend([f'{z_m}{i}', f'{x_im}{i}'])

    df = pd.DataFrame(columns=columns)


    while cap.isOpened():
        n_frames = n_frames+1
        row_data = {'n_frame': n_frames}

        #print("_")
        ret, frame = cap.read()
        if not ret:
            break


        # Calcola l'altezza dell'immagine
        height = frame.shape[0]

        # Calcola la nuova altezza dopo il ritaglio
        new_height = int(height * 0.3)  # Rimuovi un terzo dell'altezza

        # Esegui il ritaglio dell'immagine
        frame = frame[new_height:, :]

        # Trova i marker ArUco nel frame
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Esempio di regolazione del contrasto e della luminosità
        alpha = 2  # Fattore di contrasto
        beta = 5  # Fattore di luminosità
        gray_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

        # Esempio di rilevamento dei bordi con Canny edge detector




        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)





        if ids is not None:
            # Disegna i marker trovati sul framqe
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Loop attraverso i marker trovati
            # Loop attraverso i marker trovati
            #print(len(ids))
            for i in range(len(ids)):
                # Calcola la posizione 3D del marker rispetto alla telecamera
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, mtx, dist)
                marker_position = tvec[0][0]  # Posizione 3D del marker
                # Salva la posizione del marker
                x, y, z = marker_position[0], marker_position[1], marker_position[2]

                # Estrai l'ID de10l marker
                marker_id = ids[i][0]
                #print("mkr:ID", marker_id)

                # if marker_id == 1:
                #     print("x,y: ",x,y)

                # Calcola le coordinate x dei corner del marker
                x_coords = corners[i][0][:, 0]

                # Calcola la coordinata x approssimativa del centro del marker
                center_x = np.mean(x_coords)


                row_data[f'{z_m}{marker_id}'] =  z
                row_data[f'{x_im}{marker_id}'] = center_x
                row_data[f'{x_3D}{marker_id}'] = x






            # Salva il frame corrente per l'uso nel prossimo ciclo
            prev_frame = frame.copy()
            # Visualizza il frame
        cv2.imshow('Frame', resize_image(frame, 50))
        cv2.imshow("gray",resize_image(gray_image,50))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    # Rilascia le risorse
    # Salva il dataframe su un file Excel

    df.to_excel(output_excel, index=False)

    cap.release()
    cv2.destroyAllWindows()


###
def plotter_raw(df):
    # Ciclo attraverso i marker da 1 a 5
    for marker_inr in marker_ids:
        print("marker:", marker_inr)
        # Seleziona solo le righe con valori non nulli per il marker corrente
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]

        # Plot delle coordinate x e z del marker corrente come scatter plot
        plt.scatter(marker_data['n_frame'], marker_data[f'x_ip_{marker_inr}']/2000, label=f'X Coordinate Marker {marker_inr}', color='blue')
        plt.scatter(marker_data['n_frame'], marker_data[f'z_3D_{marker_inr}'], label=f'Z Coordinate Marker {marker_inr}', color='red')
        plt.scatter(marker_data['n_frame'], marker_data[f'x_3D_{marker_inr}'], label=f'X_3D Coordinate Marker {marker_inr}', color='green')

        # Aggiungi etichette e legenda al grafico corrente
        plt.xlabel('Numero Frame')
        plt.ylabel('Coordinate')
        plt.title(f'Coordinate X, X_3D e Z del Marker {marker_inr}')
        plt.legend()

        # Mostra il grafico corrente
        plt.show()

def save_results_to_excel(results, output_excel):
    # Leggi il file Excel esistente
    if os.path.exists(output_excel):
        df = pd.read_excel(output_excel)
    else:
        # Se il file non esiste, crea un DataFrame vuoto con gli header
        df = pd.DataFrame(columns=['marker', 'z_mean', 'z_std', 'vx', 'vx_std', 'vx_3D', 'vx_3D_std'])

    # Itera sui risultati e aggiungi ogni elemento del dizionario come riga nel DataFrame
    for result in results:
        df = df.append(result, ignore_index=True)

    # Salva il DataFrame aggiornato nel file Excel
    df.to_excel(output_excel, index=False)

def plotter_raw_analys(df):
    results = []  # Lista per memorizzare i risultati

    # Ciclo attraverso i marker da 1 a 5
    for marker_inr in marker_ids:
        #print("marker:", marker_inr)
        # Seleziona solo le righe con valori non nulli per il marker corrente
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]


        # Calcola il valore medio di z e l'incertezza
        z_mean = np.mean(marker_data[f'z_3D_{marker_inr}'])
        z_std = np.std(marker_data[f'z_3D_{marker_inr}'])
        try:
            if len(marker_data[f'z_3D_{marker_inr}']) > 5:
                # Calcola la velocità di x e l'incertezza utilizzando la regressioane lineare

                slope_x_ip, intercept_x_ip, _, _, std_err_x_ip = linregress(marker_data['n_frame'], marker_data[f'x_ip_{marker_inr}'])
                vx = slope_x_ip
                vx_std = std_err_x_ip

                # Calcola i valori predetti dalla regressione per x_ip
                predicted_x_ip = slope_x_ip * marker_data['n_frame'] + intercept_x_ip

                # Calcola SSE per x_ip
                SSE_x_ip = np.sum((marker_data[f'x_ip_{marker_inr}'] - predicted_x_ip) ** 2)

                # Calcola SST per x_ip
                y_mean_x_ip = np.mean(marker_data[f'x_ip_{marker_inr}'])
                SST_x_ip = np.sum((marker_data[f'x_ip_{marker_inr}'] - y_mean_x_ip) ** 2)

                # Calcola R^2 per x_ip
                R_squared_x_ip = 1 - (SSE_x_ip / SST_x_ip)
                ALL_R_2.append(R_squared_x_ip)


                # Calcola la velocità di x_3D e l'incertezza utilizzando la regressione lineare
                slope_x_3D, intercept_x_3D, _, _, std_err_x_3D = linregress(marker_data['n_frame'], marker_data[f'x_3D_{marker_inr}'])
                vx_3D = slope_x_3D * 60
                vx_3D_std = std_err_x_3D

                # Calcola le rette di regressione per x e x_3D e plotta
                x_fit = np.linspace(marker_data['n_frame'].min(), marker_data['n_frame'].max(), 100)
                y_fit_x_ip = slope_x_ip * x_fit + intercept_x_ip
                y_fit_x_3D = slope_x_3D * x_fit + intercept_x_3D


            else:
                vx = 0
                vx_3D = 0
                vx_std = 0
                vx_3D_std = 0

        except Exception as e:
            print("error regression:",e)
            vx = 0
            vx_3D = 0
            vx_std = 0
            vx_3D_std = 0

        # Aggiungi i risultati alla lista
        results.append({'marker': marker_inr, 'z_mean': z_mean, 'z_std': z_std,
                        'vx': vx, 'vx_std': vx_std, 'vx_3D': vx_3D, 'vx_3D_std': vx_3D_std})


        if PLOT_SINGLE_PATH:

        # Plot delle coordinate x e z del marker corrente come scatter plot
            plt.scatter(marker_data['n_frame'], marker_data[f'x_ip_{marker_inr}'] / 2000,
                        label=f'X Coordinate Marker {marker_inr}', color='blue')
            plt.scatter(marker_data['n_frame'], marker_data[f'z_3D_{marker_inr}'],
                        label=f'Z Coordinate Marker {marker_inr}', color='red')
            plt.scatter(marker_data['n_frame'], marker_data[f'x_3D_{marker_inr}'],
                        label=f'X_3D Coordinate Marker {marker_inr}', color='green')

            plt.plot(x_fit, y_fit_x_ip / 2000, color='green', linestyle='--', label=f'Linear Fit X Marker {marker_inr}')
            plt.plot(x_fit, y_fit_x_3D, color='orange', linestyle='--', label=f'Linear Fit X_3D Marker {marker_inr}')
            # Aggiungi etichette e legenda al grafico corrente
            plt.xlabel('Numero Frame')
            plt.ylabel('Coordinate')
            plt.title(f'Coordinate X, X_3D e Z del Marker {marker_inr}')
            plt.legend()

            # Mostra il grafico corrente
            plt.show()

    return results


#imaga_analizer_raw()
df = pd.read_excel(output_excel)
marker_rif_delation = 9
df_filtered = delete_static_data_manually(df, f'x_ip_{marker_rif_delation}', 0.02)
# plotter_raw(df_filtered)
win = 20
multi_df = smart_cutter_df(df_filtered, win)

rrr = []
coeff = []

# Calcola il passo tenendo conto della precisione
passo = 0.002

#Precisione desiderata (numero di cifre decimali significative)
precisione = 3
valori = np.arange(0.01, 0.16, passo)
valori_arrotondati = np.around(valori, decimals=precisione)
for acc_filter in valori_arrotondati:

    coeff.append(acc_filter)

    ALL_R_2 = []

    if len(multi_df) == 10:
        for i in range(len(multi_df)):
            print("dataset ",i)
            df_filtered = delete_static_data_manually(multi_df[i], f'x_ip_{marker_rif_delation}', acc_filter)
            res = plotter_raw_analys(df_filtered)
            #print(res)
            #save_results_to_excel(res, output_excel_res)
        #

    else:
        print("ERROR CUTTINGGG")


    rrr.append(np.mean(ALL_R_2))


# Crea il grafico
plt.plot(coeff, rrr)

# Aggiungi etichette agli assi
plt.xlabel('Valori di x')
plt.ylabel('Valori di y')

# Aggiungi titolo al grafico
plt.title('Grafico di x vs y')
# Adatta l'asse y ai dati
plt.gca().autoscale(axis='y')

# Mostra il grafico
plt.show()




