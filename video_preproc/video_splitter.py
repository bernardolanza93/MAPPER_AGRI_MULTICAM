import cv2
import os


# Funzione per effettuare il cropping del video
def crop_video(video_path, start_frame, end_frame, output_folder, video_count):
    # Apre il video
    cap = cv2.VideoCapture(video_path)

    # Imposta il frame iniziale
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Crea un'istanza del writer per il nuovo video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(output_folder, f'cropped_video_{video_count}.avi')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop per leggere e scrivere i frame
    for frame_index in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video {output_video_path} salvato con successo!")


# Main function
def main():
    print("main")
    video_path = 'aquisition_raw/GX010107.MP4'  # Inserisci il percorso del video
    output_folder = 'aquisition'  # Cartella in cui salvare i video croppati
    os.makedirs(output_folder, exist_ok=True)

    start_frame = 0  # Frame iniziale
    end_frame = 0  # Frame finale

    video_count = 1  # Contatore per i video

    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())

    while (cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(0)
        if key == ord('c'):  # Se viene premuto il tasto 'c', avvia il cropping
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            crop_video(video_path, start_frame, end_frame, output_folder, video_count)
            video_count += 1
            start_frame = end_frame

        elif key == ord('q'):  # Se viene premuto il tasto 'q', esce dal loop
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
