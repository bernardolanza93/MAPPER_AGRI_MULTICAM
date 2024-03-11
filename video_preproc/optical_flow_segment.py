import cv2
import numpy as np
from additional_functions import *


def calculate_derivative_flow(interpolated_flow):
    # Calcola il flusso ottico medio su tutti gli istanti temporali
    print(interpolated_flow.shape)
    mean_flow = np.mean(interpolated_flow, axis=0)

    return mean_flow

def apply_threshold(flow, threshold):
    # Calcola la magnitudine del flusso ottico
    mask = (np.abs(flow) > threshold).astype(np.uint8) * 255



    return mask

# Funzione per calcolare il flusso ottico su una finestra mobile di frame
# Funzione per calcolare il flusso ottico su una finestra mobile di frame
def calculate_optical_flow_window(frames):
    # Inizializza il flusso ottico accumulato per la finestra mobile
    accumulated_flow = np.zeros(frames[0].shape[:2] + (2,), dtype=np.float32)

    for i in range(len(frames) - 1):
        # Calcola il flusso ottico tra i frame consecutivi
        prev_frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_frame_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Aggiorna il flusso ottico accumulato
        accumulated_flow[..., 0] += flow[..., 0]
        accumulated_flow[..., 1] += flow[..., 1]

    return accumulated_flow

def generate_RGB_mask(speed_mem,flow_x):
    threshold_max = abs(speed_mem) - 1

    mask = (np.abs(flow_x) > threshold_max).astype(np.uint8) * 255

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation to connect nearby white blobs
    mask = cv2.dilate(mask, kernel, iterations=4)

    # Crea una maschera 3 canali per il frame RGB
    mask_rgb = np.stack([mask] * 3, axis=-1)

    # Applica la maschera al frame RGB
    # Crea una maschera 3 canali per il frame RGB
    mask_rgb = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Applica la maschera al frame RGB
    masked_frame = cv2.bitwise_and(previous_RGB, mask_rgb)

    return masked_frame

def generate_flow_x(gray):
    pyrScale = 0.3
    pyrLevels = 6
    winSize = 5

    polyN = 2
    polySigma = 10
    param1 = 0.5
    flow = cv2.calcOpticalFlowFarneback(previous_frame, gray, None, pyrScale, pyrLevels, winSize, polyN, polySigma,
                                        param1, 0)

    # flow = cv2.calcOpticalFlowFarneback(previous_frame, gray, None,
    #                                     flow_params['pyr_scale'], flow_params['levels'],
    #                                     flow_params['winsize'], flow_params['iterations'],
    #                                     flow_params['poly_n'], flow_params['poly_sigma'],
    #                                     flow_params['flags'])

    # Estrai la componente x del vettore di movimento
    flow_x = flow[..., 0]
    return flow_x

def contrast_and_brightness(gray):
    # Calcola il minimo e il massimo valore di intensità nel frame
    min_val = np.min(gray)
    max_val = np.max(gray)

    # Calcola l'offset e la scala per aumentare la luminosità e il contrasto
    alpha = 255.0 / (max_val - min_val)
    beta = -min_val * (255.0 / (max_val - min_val))

    # Applica la trasformazione lineare per aumentare il contrasto e la luminosità
    contrast_brightness_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    return contrast_brightness_gray


def color_filter(frame):
    # Converte l'immagine in LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Divide i canali LAB
    L, A, B = cv2.split(lab)

    # Applica un filtro sul canale A e B per migliorare il contrasto dei marroni, tralasciando il verde
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_A = clahe.apply(A)
    enhanced_B = clahe.apply(B)

    # Unisci i canali LAB modificatqi
    enhanced_lab = cv2.merge([L, enhanced_A, enhanced_B])

    # Converti l'immagine LAB modificata in BGR
    frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return frame

def calculate_mean_optical_flow_x(image, previous_frame,speed_mem):
    # Configurazione del marker ArUco
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    # Rileva i marker ArUco nell'immagine
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # Se è stato trovato almeno un marker
        marker_index = 0  # Prendi il primo marker trovato (puoi modificare questa logica se necessario)

        # Estrai i vertici del marker
        marker_corners = corners[marker_index][0]

        # Calcola il centro del marker
        marker_center = np.mean(marker_corners, axis=0)

        # Converte le coordinate del centro in coordinate intere
        marker_center_int = np.round(marker_center).astype(int)

        # Calcola l'optical flow tra due frame consecutivi dell'immagine
        flow = cv2.calcOpticalFlowFarneback(previous_frame, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Definisci la regione intorno al centro del marker dove calcolare l'optical flow
        roi_size = 20
        roi_flow = flow[max(marker_center_int[1] - roi_size, 0):min(marker_center_int[1] + roi_size, flow.shape[0]),
                        max(marker_center_int[0] - roi_size, 0):min(marker_center_int[0] + roi_size, flow.shape[1])]

        # Estrai la componente x del vettore di movimento nella regione di interesse
        flow_x = roi_flow[..., 0]

        # Calcola il valore medio dell'optical flow lungo x
        mean_flow_x = np.mean(flow_x)



        print(mean_flow_x)
        speed_mem = abs(mean_flow_x)
        return speed_mem
    else:
        print(".", end=" ")
        return speed_mem





# Leggi il video
cap = cv2.VideoCapture('/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition_raw/GX010048.MP4')

# Inizializza il calcolatore di optical flow
previous_frame = None
previous_rgb = None
speed_mem = 8
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
i = 0
while True:
    i = i + 1
    ret, frame = cap.read()
    if not ret:
        break
    print(i)
    if i > 1000:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Converti l'immagine in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        if previous_frame is not None:

            speed_mem = calculate_mean_optical_flow_x(gray, previous_frame,speed_mem)




            # Calcola l'optical flow
            flow = cv2.calcOpticalFlowFarneback(previous_frame, gray, None, 0.5, 5, 10, 3, 5, 1.2, 0)

            # Estrai la componente x del vettore di movimento
            flow_x = flow[..., 0]

            # Soglia per discriminare tra foreground e background

            mask = (np.logical_and(np.abs(flow_x) > speed_mem - 1, np.abs(flow_x) < speed_mem + 10)).astype(np.uint8) * 255
            # Define the kernel for the erosion operation
            kernel = np.ones((5, 5), np.uint8)

            # Perform erosion operation
            mask = cv2.dilate(mask, kernel, iterations=2)


            # Crea una maschera 3 canali per il frame RGB
            mask_rgb = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # Applica la maschera al frame RGB
            masked_frame = cv2.bitwise_and(previous_rgb, mask_rgb)


            # Visualizza il risultato
            zoom = 40
            cv2.imshow('Segmentation Mask', resize_image(masked_frame,zoom))
            cv2.imshow(' Mask', resize_image(previous_frame, zoom))

        previous_frame = gray
        previous_rgb = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()