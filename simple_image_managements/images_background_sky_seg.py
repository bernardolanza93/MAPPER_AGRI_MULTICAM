import os
import cv2
import numpy as np
from additional_functions import *

# Percorso della cartella di input e di output
input_folder = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition/light_first_block'
output_folder = '/home/mmt-ben/MAPPER_AGRI_MULTICAM/aquisition/processed'

# Assicurati che la cartella di output esista, altrimenti creala
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Definisci i range di colore per il cielo (blu/azzurro)
lower_blue = np.array([5, 4, 180])
upper_blue = np.array([26, 175, 255])

lower_green = np.array([75, 200, 18])
upper_green = np.array([110, 250, 80])

# Ciclo su tutti i file nella cartella di input
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Leggi l'immagine
        image = cv2.imread(os.path.join(input_folder, filename))

        # Converti l'immagine in formato HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


        # Crea una maschera per identificare il cielo
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_not(mask)

        # Riempimento dei buchi nella maschera
        # Definisci il kernel per la dilatazione
        kernel = np.ones((7, 7), np.uint8)

        # Applica la dilatazione alla maschera
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        # Applica la maschera all'immagine originale
        result = cv2.bitwise_and(image, image, mask=mask)



        cv2.imshow('Result', resize_image(hsv,30))
        cv2.imshow('jj', resize_image(result, 30))

        cv2.waitKey(1)


        # Salva l'immagine risultante nella cartella di output
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

        print(f"{filename} elaborato e salvato come {output_path}")

print("Elaborazione completata.")
cv2.destroyAllWindows()
