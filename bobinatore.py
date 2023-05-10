import cv2
import os

# cartella contenente i frame
frame_folder = "/home/mmt-ben/Documents/hokuto/"

# formato dei nomi dei file dei frame
frame_format = "frame_%d_VID_tg_.jpg"
#frame_0_VID_tg_000000.jpg

# numero di frame al secondo del video
fps = 60

# determina il numero totale di frame
num_frames = len(os.listdir(frame_folder))

# determina le dimensioni dell'immagine
img_path = os.path.join(frame_folder, frame_format % 1)
img = cv2.imread(img_path)
height, width, _ = img.shape

# crea l'oggetto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# leggi i frame, ordina in base al numero e aggiungi al video
for i in range(1, num_frames+1):
    frame_path = os.path.join(frame_folder, frame_format % i)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# chiudi il video
video_writer.release()