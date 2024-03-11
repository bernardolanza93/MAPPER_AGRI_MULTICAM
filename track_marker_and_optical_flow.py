import cv2
import numpy as np
import os
import pandas as pd
import csv
folder_calibration = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/CALIBRATION_CAMERA_FILE"
from additional_functions import *
import matplotlib.pyplot as plt


MARKER_SIZE = 0.107

# Cartella per i dati
folder_path = 'dati_marker_optical_flow'
os.makedirs(folder_path, exist_ok=True)

# Percorso del file CSV
csv_path = os.path.join(folder_path, 'dati_marker_optical_flow.csv')


def visualizza_dati(csv_path):
    # Carica i dati dal file CSV utilizzando pandas
    dati = pd.read_csv(csv_path)

    # Estrai colonne di dati
    posizione_x = dati['Posizione_X']
    posizione_y = dati['Posizione_Y']
    posizione_z = dati['Posizione_Z']
    velocita_marker_x = dati['Velocita_Marker_X']
    velocita_marker_y = dati['Velocita_Marker_Y']
    velocita_marker_z = dati['Velocita_Marker_Z']
    velocita_optical_flow_x = dati['Velocita_Optical_Flow_X']
    velocita_optical_flow_y = dati['Velocita_Optical_Flow_Y']
    MEDIA_MOBILE = 0
    media_mobile_n = 2


    # Calcola la velocità del marker utilizzando la media mobile di n valori
    if MEDIA_MOBILE:

        velocita_marker_x = np.convolve(posizione_x, np.ones(media_mobile_n)/media_mobile_n, mode='valid')
        velocita_marker_y = np.convolve(posizione_y, np.ones(media_mobile_n)/media_mobile_n, mode='valid')
        velocita_marker_z = np.convolve(posizione_z, np.ones(media_mobile_n)/media_mobile_n, mode='valid')
    else:

        # Calcola la velocità del marker in metri al secondo
        fps = 60  # Frame per secondo
        delta_t = 1 / fps  # Intervallo di tempo tra i frame
        velocita_marker_x = np.diff(posizione_x) / delta_t
        velocita_marker_y = np.diff(posizione_y) / delta_t
        velocita_marker_z = np.diff(posizione_z) / delta_t

        # Aggiungi zero per compensare la perdita di un elemento dopo np.diff
        velocita_marker_x_or = np.concatenate(([0], velocita_marker_x))
        velocita_marker_y_or = np.concatenate(([0], velocita_marker_y))
        velocita_marker_z_or = np.concatenate(([0], velocita_marker_z))

        # Applica un filtro di media mobile alla velocità del marker
        window_size = 25  # Dimensione della finestra per il filtro di media mobile
        velocita_marker_x = np.convolve(velocita_marker_x_or, np.ones(window_size) / window_size, mode='same')
        velocita_marker_y = np.convolve(velocita_marker_y_or, np.ones(window_size) / window_size, mode='same')
        velocita_marker_z = np.convolve(velocita_marker_z_or, np.ones(window_size) / window_size, mode='same')

    # Grafico della posizione
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(posizione_x, label='X')
    plt.plot(posizione_y, label='Y')
    plt.plot(posizione_z, label='Z')
    plt.title('Posizione del marker rispetto alla telecamera')
    plt.legend()

    # Grafico della velocità del marker
    plt.subplot(3, 1, 2)
    plt.plot(velocita_marker_x_or, label='Velocità Marker X')
    #plt.plot(velocita_marker_x_or, label='Velocità Marker X ORIGINALE')
    #plt.plot(velocita_marker_y, label='Velocità Marker Y')
    #plt.plot(velocita_marker_z, label='Velocità Marker Z')
    plt.title('Velocità del marker rispetto alla telecamera (Media Mobile)')
    plt.legend()

    # Grafico della velocità dell'optical flow
    plt.subplot(3, 1, 3)
    plt.plot(velocita_optical_flow_x, label='Velocità Optical Flow X')
    plt.plot(velocita_optical_flow_y, label='Velocità Optical Flow Y')
    plt.title('Velocità dell\'optical flow rispetto al marker')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calcola_velocita_movimento(video_path):
    # Carica i parametri di calibrazione della camera
    mtx = np.load(os.path.join(folder_calibration, "camera_matrix.npy"))
    dist = np.load(os.path.join(folder_calibration, "dist_coeffs.npy"))

    # Carica il video
    cap = cv2.VideoCapture(video_path)

    # Inizializza il detector di marker ArUco
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    # Inizializza il rilevatore di feature per l'optical flow
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_frame = None
    prev_corners = None
    # Array per salvare i dati
    dati = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rileva il marker ArUco nel frame
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None:
            # Calcola la velocità di movimento rispetto al marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
            velocita_movimento_camera_marker = np.linalg.norm(tvec[0][0])

            # Calcola la profondità del marker rispetto alla telecamera
            profondita_marker = tvec[0][0][2]

            # Utilizza gli spigoli del marker come punti salienti per l'optical flow
            marker_points = corners[0].astype(np.float32).reshape(-1, 2)

            # Calcola l'optical flow solo sui punti del marker
            if prev_frame is not None and prev_corners is not None:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_corners, None, **lk_params)

                # Se ci sono abbastanza punti validi, calcola la velocità dell'optical flow
                valid_points = status.flatten() == 1
                if np.sum(valid_points) > 0:
                    new_valid_points = new_points[valid_points]
                    prev_valid_corners = prev_corners[valid_points]
                    velocita_optical_flow_x = (new_valid_points[:, 0] - prev_valid_corners[:, 0])
                    velocita_optical_flow_y = (new_valid_points[:, 1] - prev_valid_corners[:, 1])
                    velocita_optical_flow_marker = np.mean(
                        np.linalg.norm(new_valid_points - prev_valid_corners, axis=-1))

                    # Calcola le componenti della velocità rispetto al marker
                    velocita_x = (new_valid_points[:, 0] - prev_valid_corners[:,
                                                           0]) / 1.0  # 1.0 è la lunghezza del marker
                    velocita_y = (new_valid_points[:, 1] - prev_valid_corners[:, 1]) / 1.0
                    velocita_z = np.zeros_like(
                        velocita_x)  # Non possiamo calcolare velocità Z senza la lunghezza del marker

                    # Salva i dati in un array
                    dati.append([tvec[0][0][0], tvec[0][0][1], tvec[0][0][2],
                                 velocita_x.mean(), velocita_y.mean(), velocita_z.mean(),
                                 velocita_optical_flow_x.mean(), velocita_optical_flow_y.mean()])

                    # Disegna i punti del marker
                    #frame = cv2.aruco.drawDetectedMarkers(frame, corners)

                    # Disegna la linea dell'optical flow
                    for i, (new, old) in enumerate(zip(new_valid_points, prev_valid_corners)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        frame = cv2.line(frame, (a, b), (c, d), (255, 0, 0), 1)
                        frame = cv2.circle(frame, (a, b), 8, (0, 255, 0), 1)


                    # Stampa le misure scomponendole lungo gli assi del marker
                    print(f"Velocità camera rispetto al marker: {velocita_movimento_camera_marker}")
                    print(f"Profondità del marker rispetto alla telecamera: {profondita_marker}")
                    print(f"Velocità dell'optical flow rispetto al marker: {velocita_optical_flow_marker}")

            # Aggiorna i frame precedenti e i punti per il prossimo ciclo
            prev_frame = gray.copy()
            prev_corners = marker_points

        # Mostra l'immagine con l'optical flow
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Optical Flow', resize_image(frame,50))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()
    # Converti l'array dei dati in un array NumPy
    dati = np.array(dati)

    # Salva i dati in un file CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Posizione_X', 'Posizione_Y', 'Posizione_Z',
                            'Velocita_Marker_X', 'Velocita_Marker_Y', 'Velocita_Marker_Z',
                            'Velocita_Optical_Flow_X', 'Velocita_Optical_Flow_Y'])
        csvwriter.writerows(dati)
    # Visualizza i dati utilizzando Matplotlib




# Esempio di utilizzo
video_path = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/GX010091.MP4'

#calcola_velocita_movimento(video_path)

visualizza_dati(csv_path)
