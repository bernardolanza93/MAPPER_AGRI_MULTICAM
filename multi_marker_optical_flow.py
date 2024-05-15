import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
from additional_functions import *
from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import norm
from itertools import groupby
from statistics import mean



# Salva il dataframe su un file Excel
output_excel = 'marker_data.xlsx'

###modella tutte le curve e estrai un k (5 k) poi graficali rispetto a vext, e quindi puoi stimare k conoscendo la V
#proseguire con il modello ottico cercando meglio.
#non prendere media e dev std ma servono tutti i valori di velocità per prova e distanzaq

# Definisci il percorso della cartella di calibrazione
folder_calibration = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/CALIBRATION_CAMERA_FILE"
PLOT_SINGLE_PATH = 0
# Carica i parametri di calibrazione della camera
mtx = np.load(os.path.join(folder_calibration, "camera_matrix.npy"))
dist = np.load(os.path.join(folder_calibration, "dist_coeffs.npy"))
print("mtx",mtx)

# Estrai i parametri dalla matrice di calibrazione
fx = mtx[0, 0]  # Lunghezza focale lungo l'asse x
fy = mtx[1, 1]  # Lunghezza focale lungo l'asse y
cx = mtx[0, 2]  # Coordinata x del centro di proiezione
cy = mtx[1, 2]  # Coordinata y del centro di proiezione

video_name = 'GX010118.MP4'
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
v1 = 0.25
v2 = 0.5
v3 = 0.75
v4 = 0.94
v5 = 0.97
v_all = [v1,v2,v3,v4,v5]



# Nome del file Excel
output_excel_res = 'results.xlsx'
if os.path.exists(output_excel_res):

    print("file esiste appendo res")


else:

    # Header del file Excel
    header = ['timestamp','7_z', '7_vx','8_z', '8_vx','9_z', '9_vx','10_z', '10_vx','11_z', '11_vx']

    # Crea un DataFrame vuoto con gli header
    df_res = pd.DataFrame(columns=header)

    # Salva il DataFrame con gli header nel file Excel
    df_res.to_excel(output_excel_res, index=False)

    print(f"Creato il file '{output_excel}' con gli header vuoti.")
# Funzione per il salvataggio dei risultati nel file constant


def synchro_data_v_v_e_z(file_raw_optics):

    data = pd.read_csv("data/1b.csv", delimiter=",")

    min_timestamp = data['__time'].min()
    # Estrai i dati filtrati
    timestamps = data['__time']

    # Converti il timestamp in secondi sottraendo il minimo timestamp e dividendo per 1 secondo
    timestamps_in_seconds = (data['__time'] - min_timestamp) / 1.0  # 1 secondo

    # Plot dei dati di posizione come scatter plot
    plt.scatter(timestamps_in_seconds, data['/tf/base/tool0_controller/translation/x'], label='Position Data', s=10)

    # Aggiungi titoli e legenda
    plt.title(f'Robot trajectories')
    plt.xlabel('time [s]')
    plt.ylabel('Traslazione X [m]')
    plt.legend()

    plt.show()


def media_mobile(lista, window_size):
    """
    Calcola la media mobile di una lista data una finestra di dimensione window_size.

    :param lista: La lista di valori.
    :param window_size: La dimensione della finestra per il calcolo della media mobile.
    :return: La lista dei valori della media mobile.
    """
    lista = np.array(lista)
    padding = window_size // 2  # Calcolo del padding necessario per mantenere la stessa lunghezza dell'input
    lista_padded = np.pad(lista, (padding, padding), mode='edge')  # Padding con il primo/ultimo valore per mantenere la stessa lunghezza
    moving_avg = np.convolve(lista_padded, np.ones(window_size) / window_size, mode='valid')
    return moving_avg[:len(lista)]  # Rimuove gli elementi in eccesso per mantenere la stessa lunghezza dell'input


def smoothing(x_fps, marker_n, window_size):
    data = x_fps
    """
      Funzione che suddivide una lista in sottoliste basandosi sul cambio di marker ID e le riassembla mantenendo l'ordine originale.

      Argomenti:
        data: Lista di dati da suddividere e riassemblare.
        marker_n: Lista di marker ID corrispondenti ai dati.

      Restituisce:
        Lista riassemblata con i dati in ordine originale.
      """

    sublists = []  # Lista per memorizzare le sottoliste
    current_sublist = []  # Lista temporanea per la sottolista corrente
    current_marker = None  # Marker ID corrente

    for i, (datum, marker) in enumerate(zip(data, marker_n)):
        # Controlla il cambio di marker
        if current_marker != marker:

            #print("Spezzato")
            #modifica sublist
            #print(current_sublist)
            #print(current_sublist)
            if len(current_sublist) > window_size:

                print(len(current_sublist))
                current_sublist = media_mobile(current_sublist, window_size)
                print(len(current_sublist))


            sublists.append(current_sublist)
            current_sublist = []
            current_marker = marker

        # Aggiungi il dato alla sottolista corrente
        current_sublist.append(datum)

    # Gestisce l'ultima sottolista (se presente)
    if current_sublist:
        print("Spezzato")
        sublists.append(current_sublist)

    # Riassembla i dati in ordine originale
    reassembled_data = []
    for sublist in sublists:
        reassembled_data.extend(sublist)

    return reassembled_data


def hist_adv(residui):
    # Calcola l'errore sistematico
    errore_sistematico = np.mean(residui)

    # Calcola l'errore casuale
    errore_casuale = np.std(residui)

    # Plotta l'istogramma dei residui
    plt.hist(residui, bins=30, color='skyblue', edgecolor='black', density=True, alpha=0.6)

    # Calcola la deviazione standard della distribuzione gaussiana
    sigma_standard = np.std(residui)

    # Crea un array di valori x per la distribuzione gaussiana
    x_gauss = np.linspace(np.min(residui), np.max(residui), 100)

    # Calcola i valori y corrispondenti alla distribuzione gaussiana
    y_gauss = norm.pdf(x_gauss, np.mean(residui), np.std(residui))

    # Plotta la distribuzione gaussiana sopra l'istogramma dei residui
    plt.plot(x_gauss, y_gauss, 'r--', label='Distribuzione Gaussiana')

    # Plotta le linee verticali corrispondenti alla deviazione standard
    plt.axvline(x=errore_sistematico + errore_casuale, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=errore_sistematico - errore_casuale, color='k', linestyle='--', linewidth=1)
    # Aggiungi una linea verticale corrispondente al valore medio dei residui
    plt.axvline(x=np.mean(residui), color='g', linestyle='-', linewidth=3)

    # Aggiungi la deviazione standard nel titolo
    plt.title(f'Istogramma dei Residui\nDeviazione Standard: {sigma_standard:.4f}')

    plt.xlabel('Residui [m]')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_outlier(x,y):
    # Converti le serie Pandas in array NumPy
    x = np.array(x)
    y = np.array(y)

    # Calcola il primo e il terzo quartile di x e y
    Q1_x, Q3_x = np.percentile(x, [10 ,90])
    Q1_y, Q3_y = np.percentile(y, [10, 90])

    # Calcola l'interquartile range di x e y
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y

    # Definisci il range per considerare un valore outlier
    range_outlier = 1.5

    # Trova gli outlier in x
    outlier_x = (x < Q1_x - range_outlier * IQR_x) | (x > Q3_x + range_outlier * IQR_x)

    # Trova gli outlier in y
    outlier_y = (y < Q1_y - range_outlier * IQR_y) | (y > Q3_y + range_outlier * IQR_y)

    # Unisci gli outlier trovati sia in x che in y
    outlier = outlier_x | outlier_y

    # Rimuovi gli outlier da x e y
    x_filtrato = x[~outlier]
    y_filtrato = y[~outlier]

    # Stampa il numero di outlier rimossi
    numero_outlier_rimossi = np.sum(outlier)
    print(f"Hai rimosso {numero_outlier_rimossi} outlier.")
    return x_filtrato , y_filtrato





def save_to_file_OF_results(filename, constant, constant_uncert, velocity):
    with open(filename, 'a') as file:
        file.write(f"{constant},{constant_uncert},{velocity}\n")

# Funzione per fare regressione lineare con incertezza
def weighted_linregress_with_error_on_y(x, y, y_err):
    # Pesi basati sull'errore sull'asse y
    w = 1 / y_err

    # Calcola la media pesata dei valori
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)

    # Calcola le covarianze pesate
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    cov_xx = np.sum(w * (x - x_mean) ** 2)

    # Calcola il coefficiente di regressione pesato e l'intercetta
    slope = cov_xy / cov_xx
    intercept = y_mean - slope * x_mean

    # Calcola l'R^2 considerando solo l'errore sulle y
    residui = y - (slope * x + intercept)
    somma_quadri_residui = np.sum(w * residui ** 2)
    totale = np.sum(w * (y - y_mean) ** 2)
    r_squared = 1 - (somma_quadri_residui / totale)

    return slope, intercept, r_squared

def modello(x, costante):
    return costante / x

def compute_dz(Vx, Vx_prime, fx, fy, cx, cy):
    dz = ((Vx * fx) / Vx_prime )

    return dz



def windowing_vs_uncertanty(file_path):
    SHOW_PLOT = 0

    v_ext = []
    unc_k = []
    sigma_gauss = []
    win_size = []

    for window_size in range(1,10,1):


        # Elimina il file constant se esiste
        if os.path.exists("constant.txt"):
            os.remove("constant.txt")

        # Crea il file constant con gli header
        with open("constant.txt", 'w') as file:
            file.write("constant,constant_uncert,velocity\n")


        data = pd.read_excel(file_path)
        # Rendi positivi i valori di vx e vx_std

        # Rendi positivi i valori di vx e vx_std
        data['vx'] = abs(data['vx'])


        # Rimuovi le righe con zeri o valori mancanti nella riga
        data = data[(data != 0).all(1)]

        # Dividi il DataFrame in base al valore della colonna vx_3D
        gruppi = data.groupby('vx_3D')

        # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
        sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}

        for chiave, valore in sotto_dataframe.items():
            print(chiave, valore)
            data = sotto_dataframe[chiave]



            # Definisci i colori per i diversi valori di vx_3D
            color_map = {
                v1: 'red',
                v2: 'blue',
                v3: 'green',
                v4: 'orange',
                v5: 'purple'
            }
            print(fx,fy,cx,cy)



            x_fps = data['vx']





            marker_n = data['marker']
            x = [element * 60 for element in x_fps]


            y = data['z_mean']

            SMOOTHING = 1
            window = 0


            if SMOOTHING:
                window = 7
                x_or = x
                x_s = smoothing(x,marker_n, window_size)
                x_s_graph = [x_ii + 1000 for x_ii in x_s]
                x = x_s

            #x = media_mobile(x,150)


            color_p = color_map[chiave]

            PLOT_OF_RAW  = 1
            if PLOT_OF_RAW and SMOOTHING:


                x__1 = list(range(len(x)))
                #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
                plt.plot(x__1, x_or)
                plt.plot(x__1, x_s_graph)
                marker_aug =  [element * 100 for element in marker_n]
                plt.plot(x__1,marker_aug)
                if SHOW_PLOT:
                    plt.show()



            # Vx_prime_values = np.linspace(min(x), max(x), 100)
            Vx_prime_values = sorted(x)


            # Adatta il modello ai dati
            parametri, covarianza = curve_fit(modello, x, y)


            # Estrai la costante stimata
            costante_stimata = parametri[0]

            # Calcola l'incertezza associata alla costante
            incertezza_costante = np.sqrt(np.diag(covarianza))[0]

            # Calcola l'R^2
            residui = y - modello(x, costante_stimata)
            somma_quadri_residui = np.sum(residui ** 2)
            totale = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (somma_quadri_residui / totale)

            # Calcola i valori del modello per il plotting
            x_modello = np.linspace(min(x), max(x), 100)
            y_modello = modello(x_modello, costante_stimata)

            # Salva i dati nel file
            save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)

            plt.figure(figsize=(15, 10))

            # Grafico dei punti grezzi e del modello
            plt.scatter(x, y, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")



            #MODELLO GENERICO


            plt.plot(x_modello, y_modello, label='Modello genereico Dz = k/OF', color='black',linestyle='-.',)

            plt.xlabel('OF [px/s]')
            plt.ylabel('Depth [m]')
            plt.grid(True)
            plt.ylim(0, 2.1)

            # Plot aggiunto
            Y_teorico = []
            for i in range(len(Vx_prime_values)):


                dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
                Y_teorico.append(dzi)

            plt.plot(Vx_prime_values, Y_teorico, color="grey",label='Modello teorico Dz = (V_r * fx)/OF')


            # Calcola l'errore sistematico
            residui = (y - Y_teorico) / y
            errore_sistematico = np.mean(residui)

            # Calcola l'errore casuale
            errore_casuale = np.std(residui)
            # Calcola i residui

            costante_teorica = fx * float(chiave)

            plt.title(
                f'depth vs Optical flow [z = k / vx] - media mobile filtro :{window}, \n K_th: {costante_teorica:.2f} , K_exp:{costante_stimata:.2f} +- {incertezza_costante:.2f} [px*m]  o [px * m/s] || R^2:{r_squared:.4f} \n Stat on relative residuals (asimptotic - no gaussian): \n epsilon_sistem_REL :  {errore_sistematico*100 :.3f}% , sigma_REL: {errore_casuale*100 :.3f} %')


            # Posiziona la legenda in alto a destra
            plt.legend(loc="upper right")




            # Percorso del file di salvataggio
            file_path_fig = 'results/speed_'+ str(chiave) +'_k_model.png'

            # Verifica se il file esiste già
            if os.path.exists(file_path_fig):
                # Se il file esiste, eliminilo
                os.remove(file_path_fig)
                print("removed old plot")

            # Salva la figura
            plt.savefig(file_path_fig)


            if SHOW_PLOT:
                plt.show()

            if SHOW_PLOT:

                hist_adv(residui)

            v_ext.append(color_p)
            unc_k.append(incertezza_costante)
            sigma_gauss.append(errore_casuale)
            win_size.append(window_size)


    plt.close('all')
    # Creazione dei subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Grafico 1: Incertezza associata ai parametri del modello
    for i in range(len(v_ext)):
        ax1.scatter(win_size[i], unc_k[i], color=v_ext[i], marker="x", label='Model ' + str(i + 1))
    ax1.set_xlabel('Window Size [samples]')
    ax1.set_ylabel('k uncertanty [m*px]')


    # Grafico 2: Sigma del modello fittato (Sigma Gauss)
    for i in range(len(v_ext)):
        ax2.scatter(win_size[i], sigma_gauss[i], color=v_ext[i], label='Model ' + str(i + 1))
    ax2.set_xlabel('Window Size [samples]')
    ax2.set_ylabel('relative sigma of residuals [std]')

    # Imposta il titolo del subplot
    fig.suptitle('Model Evaluation - moving avarege effect')

    # Mostra il plot
    plt.show()


def show_result_ex_file(file_path):
    SHOW_PLOT = 1


    # Elimina il file constant se esiste
    if os.path.exists("constant.txt"):
        os.remove("constant.txt")

    # Crea il file constant con gli header
    with open("constant.txt", 'w') as file:
        file.write("constant,constant_uncert,velocity\n")


    data = pd.read_excel(file_path)
    # Rendi positivi i valori di vx e vx_std

    # Rendi positivi i valori di vx e vx_std
    data['vx'] = abs(data['vx'])


    # Rimuovi le righe con zeri o valori mancanti nella riga
    data = data[(data != 0).all(1)]

    # Dividi il DataFrame in base al valore della colonna vx_3D
    gruppi = data.groupby('vx_3D')

    # Crea un dizionario di sotto-dataframe, dove ogni chiave è un valore univoco di vx_3D
    sotto_dataframe = {key: gruppi.get_group(key) for key in gruppi.groups}

    for chiave, valore in sotto_dataframe.items():
        print(chiave, valore)
        data = sotto_dataframe[chiave]



        # Definisci i colori per i diversi valori di vx_3D
        color_map = {
            v1: 'red',
            v2: 'blue',
            v3: 'green',
            v4: 'orange',
            v5: 'purple'
        }
        print(fx,fy,cx,cy)



        x_fps = data['vx']





        marker_n = data['marker']
        x = [element * 60 for element in x_fps]


        y = data['z_mean']

        SMOOTHING = 1
        window = 0


        if SMOOTHING:
            window = 7
            x_or = x
            x_s = smoothing(x,marker_n, window)
            x_s_graph = [x_ii + 1000 for x_ii in x_s]
            x = x_s

        #x = media_mobile(x,150)


        color_p = color_map[chiave]

        PLOT_OF_RAW  = 1
        if PLOT_OF_RAW and SMOOTHING:


            x__1 = list(range(len(x)))
            #plt.scatter(x__1, x, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")
            plt.plot(x__1, x_or)
            plt.plot(x__1, x_s_graph)
            marker_aug =  [element * 100 for element in marker_n]
            plt.plot(x__1,marker_aug)
            if SHOW_PLOT:
                plt.show()



        # Vx_prime_values = np.linspace(min(x), max(x), 100)
        Vx_prime_values = sorted(x)


        # Adatta il modello ai dati
        parametri, covarianza = curve_fit(modello, x, y)


        # Estrai la costante stimata
        costante_stimata = parametri[0]

        # Calcola l'incertezza associata alla costante
        incertezza_costante = np.sqrt(np.diag(covarianza))[0]

        # Calcola l'R^2
        residui = y - modello(x, costante_stimata)
        somma_quadri_residui = np.sum(residui ** 2)
        totale = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (somma_quadri_residui / totale)

        # Calcola i valori del modello per il plotting
        x_modello = np.linspace(min(x), max(x), 100)
        y_modello = modello(x_modello, costante_stimata)

        # Salva i dati nel file
        save_to_file_OF_results("constant.txt", costante_stimata, incertezza_costante, chiave)

        plt.figure(figsize=(15, 10))

        # Grafico dei punti grezzi e del modello
        plt.scatter(x, y, label='Dati raw', color=color_p, s=35,alpha=0.05,marker ="o",edgecolor ="black")



        #MODELLO GENERICO


        plt.plot(x_modello, y_modello, label='Modello genereico Dz = k/OF', color='black',linestyle='-.',)

        plt.xlabel('OF [px/s]')
        plt.ylabel('Depth [m]')
        plt.grid(True)
        plt.ylim(0, 2.1)

        # Plot aggiunto
        Y_teorico = []
        for i in range(len(Vx_prime_values)):


            dzi = compute_dz(float(chiave), Vx_prime_values[i], fx, fy, cx, cy)
            Y_teorico.append(dzi)

        plt.plot(Vx_prime_values, Y_teorico, color="grey",label='Modello teorico Dz = (V_r * fx)/OF')


        # Calcola l'errore sistematico
        residui = (y - Y_teorico) / y
        errore_sistematico = np.mean(residui)

        # Calcola l'errore casuale
        errore_casuale = np.std(residui)
        # Calcola i residui

        costante_teorica = fx * float(chiave)

        plt.title(
            f'depth vs Optical flow [z = k / vx] - media mobile filtro :{window}, \n K_th: {costante_teorica:.2f} , K_exp:{costante_stimata:.2f} +- {incertezza_costante:.2f} [px*m]  o [px * m/s] || R^2:{r_squared:.4f} \n Stat on relative residuals (asimptotic - no gaussian): \n epsilon_sistem_REL :  {errore_sistematico*100 :.3f}% , sigma_REL: {errore_casuale*100 :.3f} %')


        # Posiziona la legenda in alto a destra
        plt.legend(loc="upper right")




        # Percorso del file di salvataggio
        file_path_fig = 'results/speed_'+ str(chiave) +'_k_model.png'

        # Verifica se il file esiste già
        if os.path.exists(file_path_fig):
            # Se il file esiste, eliminilo
            os.remove(file_path_fig)
            print("removed old plot")

        # Salva la figura
        plt.savefig(file_path_fig)


        if SHOW_PLOT:
            plt.show()

        if SHOW_PLOT:

            hist_adv(residui)






def constant_analisis():
    # Leggi i dati dal file
    data = np.loadtxt("constant.txt", delimiter=',', skiprows=1)

    # Estrai le colonne
    constant_data = data[:, 0]
    constant_uncert_data = data[:, 1]
    velocity_data = data[:, 2]

    # Fai la regressione lineare tenendo conto dell'incertezza sulla costante
    slope, intercept, r_squared = weighted_linregress_with_error_on_y(velocity_data, constant_data, 1 / constant_uncert_data)
    # Calcola l'incertezza della pendenza
    residuals = constant_data - (slope * velocity_data + intercept)
    uncert_slope = np.sqrt(np.sum(constant_uncert_data ** 2 * residuals ** 2) / np.sum((velocity_data - np.mean(velocity_data)) ** 2))
    # Calcola l'R^2

    sigma3 = [element * 3 for element in constant_uncert_data]

    plt.figure(figsize=(12, 7))
    # Grafico
    plt.scatter(velocity_data, constant_data, label='Dati',s = 15)
    plt.errorbar(velocity_data, constant_data, yerr=sigma3, fmt='none', label='Incertezza')
    plt.plot(velocity_data, slope * velocity_data + intercept, color='red', label='k(v_ext) sperimentale')
    plt.plot(velocity_data, velocity_data* fx, color='orange', label='K(v_ext) teorico')
    plt.xlabel('V_ext[m/s]')
    plt.ylabel('Constant [k]')
    plt.title(
        f' k_i = f(v_ext) : slope:{slope:.1f} sigma:{uncert_slope:.1f} k/[m/s]|| R^2:{r_squared:.4f} \n incertezza su parametri: {constant_uncert_data[0]:.2f} , {constant_uncert_data[1]:.2f},{constant_uncert_data[2]:.2f},{constant_uncert_data[3]:.2f} [px*m] - 99.7% int')
    plt.legend()
    plt.grid(True)



    # Percorso del file di salvataggio
    file_path_fig = 'results/k_LR.png'

    # Verifica se il file esiste già
    if os.path.exists(file_path_fig):
        # Se il file esiste, eliminilo
        os.remove(file_path_fig)
        print("removed old plot")

    # Salva la figura
    plt.savefig(file_path_fig)
    plt.show()


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
    PROC = 1



    n_frames = 0

    if PROC:
        # Elimina il file se già esiste
        if os.path.exists(output_excel):
            os.remove(output_excel)

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
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        row_data = {'timestamp': timestamp}
        if not ret:
            break




        if PROC:
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

        if PROC:




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
                    z_key = f'{marker_id}_z'
                    vx_key = f'{marker_id}_vx'

                    row_data[z_key] = z
                    row_data[vx_key] = center_x







            # Salva il frame corrente per l'uso nel prossimo ciclo
            prev_frame = frame.copy()
            # Visualizza il frame
        cv2.imshow('Frame', resize_image(frame, 50))
        #cv2.imshow("gray",resize_image(gray_image,50))
        cv2.imshow("hhh", resize_image(gray_image, 50))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if PROC:
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    # Rilascia le risorse
    # Salva il dataframe su un file Excel
    if PROC:
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
    # Leggi il file Excel esistente, se presente
    if os.path.exists(output_excel):
        df = pd.read_excel(output_excel)
    else:
        df = pd.DataFrame()  # Crea un nuovo DataFrame se il file Excel non esiste

    # Itera sui risultati e aggiungi ogni elemento del dizionario come riga nel DataFrame
    for result in results:
        dict_to_add = {}
        # Estrai il valore per ogni colonna dal dizionario e aggiungi come riga al DataFrame
        for key, value_list in result.items():
            # Verifica se la chiave (header) esiste già nel DataFrame
            if key in df.columns:
                # Se la colonna esiste già, estendi la Serie esistente con i nuovi valori

                dict_to_add[key] = value_list
        df = df.append(pd.DataFrame(dict_to_add))



    # Salva il DataFrame aggiornato nel file Excel
    df.to_excel(output_excel, index=False)
def plotter_raw_analys(df,v_rob):
    results = []  # Lista per memorizzare i risultati

    # Ciclo attraverso i marker da 1 a 5
    for marker_inr in marker_ids:
        #print("marker:", marker_inr)
        # Seleziona solo le righe con valori non nulli per il marker corrente
        marker_data = df[df[f'x_ip_{marker_inr}'].notnull() & df[f'z_3D_{marker_inr}'].notnull()]



        all_z_depth = marker_data[f'z_3D_{marker_inr}'].tolist()

        posizione = marker_data[f'x_ip_{marker_inr}'].tolist()


        # Calcola i delta tra i valori di posizione successivi
        opt_flow = [posizione[i + 1] - posizione[i] for i in range(len(posizione) - 1)]


        # Estendi la lista dei delta in modo che abbia la stessa dimensione della lista di input di posizione
        opt_flow.append(opt_flow[-1])  # estendi con l'ultimo valore


        marker_list = [int(marker_inr)] * len(all_z_depth)
        speed_rob_list = [v_rob] * len(all_z_depth)
        # Aggiungi i risultati alla lista
        results.append({'marker': marker_list, 'z_mean': all_z_depth,'vx': opt_flow, 'vx_3D': speed_rob_list})




    return results
#
# #
# imaga_analizer_raw()
# df = pd.read_excel(output_excel)
# marker_rif_delation = 9
# df_filtered = delete_static_data_manually(df, f'x_ip_{marker_rif_delation}', 0.04)
# #plotter_raw(df_filtered)
# win = 20
# multi_df = smart_cutter_df(df_filtered, win)
#
# acc_filter_final = 0.08
#
#
#
#
# if len(multi_df) == 10:
#     ALL_R_2 = []
#     for i in range(len(multi_df)):
#         v_i = i // 2
#         speed_rob = v_all[v_i]
#         print(f"dts = {i}, vct = {speed_rob}")
#
#         df_filtered = delete_static_data_manually(multi_df[i], f'x_ip_{marker_rif_delation}', acc_filter_final)
#         res = plotter_raw_analys(df_filtered,speed_rob)
#         #print(res)
#         save_results_to_excel(res, output_excel_res)
#
#     #
#
# else:
#     print("ERROR CUTTINGGG")

# file_path = 'dati_of/stich_grande.xlsx'
# show_result_ex_file(file_path)


synchro_data_v_v_e_z("results_raw.xlsx")


sys.exit()
file_path_1 = 'dati_of/all_points_big_fix_speed.xlsx'
show_result_ex_file(file_path_1)
windowing_vs_uncertanty(file_path_1)


constant_analisis()




#
#
#
#
#
# rrr = []
# coeff = []
#
# # Calcola il passo tenendo conto della precisione
# passo = 0.002
#
# #Precisione desiderata (numero di cifre decimali significative)
# precisione = 3
# valori = np.arange(0.01, 0.16, passo)
# valori_arrotondati = np.around(valori, decimals=precisione)
# acc_filter_final = 0.075
# for acc_filter in valori_arrotondati:
#
#     coeff.append(acc_filter)
#
#     ALL_R_2 = []
#
#     if len(multi_df) == 10:
#         for i in range(len(multi_df)):
#             print("dataset ",i)
#             df_filtered = delete_static_data_manually(multi_df[i], f'x_ip_{marker_rif_delation}', acc_filter)
#             res = plotter_raw_analys(df_filtered)
#             #print(res)
#             #save_results_to_excel(res, output_excel_res)
#         #
#
#     else:
#         print("ERROR CUTTINGGG")
#
#
#     rrr.append(np.mean(ALL_R_2))
#
#
# # Crea il grafico
# plt.plot(coeff, rrr)
#
# # Aggiungi etichette agli assi
# plt.xlabel('Valori di x')
# plt.ylabel('Valori di y')
#
# # Aggiungi titolo al grafico
# plt.title('Grafico di x vs y')
# # Adatta l'asse y ai dati
# plt.gca().autoscale(axis='y')
#
# # Mostra il grafico
# plt.show()
