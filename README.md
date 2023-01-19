

# MAPPER_AGRI_MULTICAM

acquisition and mapping system for agricoltural infield analisys.
Programma per valutare l'indice di crescita della vite tramite sistema di visione.

## Description

Questa repository contiene sia il software di acquisizione che il software di analisi per delle misure ottiche in agricoltura. 
Il file main.py permette di eseguire un programma in grado di acquisire e salvare immagini provenienti da una camera INTEL Realsense D435, tramita una scheda NVIDIA Jetson Nano.
Il file media_mapper_evaluator e il relativo file evaluator_utils (contenente le funzioni innestatate) permetto l analisi dei media prodotti dal main, analisi che viene condotta su PC. Siamo quinid in grado di produrre misure geometriche e volumetriche a partire daalle acquisizioni e stimare la biomassa lignea presente all interno delle nostre acquisizioni.

#Configurazione sensori in campo  

La telecamera va posizionata ad una distanza fissa dal filare, in modo che inquadri completamente l’area di potatura. È necessario stimare:
-La fascia di potatura (asse z dalla fine del tronco alla massima altezza di sviluppo dei rami)
-FOV verticale della camera D435i 
La distanza dal filare dipenderà da questi due parametri ( Df ~ Hfov , Lfp)

#Protocollo salvataggio dati

Vengono salvati i dati Row tramite due matrici di streaming video, una RGB (3 canali) e una DEPTH ( 3 canalo da riconvertire poi in depth monocalnale). Il labed del video sarà 
-timestampato (darà l’ordine di lettura [es min_sec_millsec]) e
-con le coordinate x,y,z provenienti dalla camera T265 (x_y_z) salvate su un file csv a parte.
-Per utilizzare i dati della T265 è necessario creare un sistema di riferimento assoluto tramite un ARUCO MARKER per conoscere la posizione di partenza con un ottima accuratezza

#Elaborazione a segmentazione

Vengono segmentati i dati acquisiti in laboratorio su background fisso (senza T265).
Viene estratta una maschera cromatica B&W monocanale contenete i pixel volume ligneo.

#Analisi dati

Vengono effettuate delle misurazioni di volume (a partire dalla calibrazione statica pixel/volume con campioni fermi ed equidistanti dall’ottica). Saranno effettuate delle misure volumetriche tramite serbatoio graduato per conoscere il volume vero dei campioni. A partire da un acquisizione statica e confronto con il riferimento si valuteranno i valori di RMS. Valuteremo anche la STD-DEV valutando come la misura di influenzata da effetti luminosi in campo statico. 
Queste prove saranno estese a condizioni più critiche, come:
-la diminuzione della risoluzione del misurando (allontanamento il campione dalla camera) individuazione della distanza misurando camera ottimale
-Aumento della variabilità della profondità del misurando (campione orientato in posizioni complesse a differenti profondità)

#Protocollo labializzazione dati

DATASET AUGMENTATION
Utilizzando le acquisizioni row in lab a bg fisso di uno o più tralci si utilizzerà la maschera ottenuta per generare il label di Training DNN. Si utilizzeranno come input delle acquisizioni post potatura con incollata l’acquisizione in lab (stile fotomontaggio) dei soli pixel rgb segmentati.
DATASET CREATION 
si userà come input le acquisizioni row in filare, la differenza con le immagini del post potato segmentato ci darà il label per la DNN
Faremo tante acquisizioni, a camera fissa, in cui prima metteremo un telo bianco di sfondo, e poi riacquisiremo senza telo in modo da avere info di segmentazione automatiche (il video puo continuare ad andare per evitare di muovere la camera)
DATASET FROM DARK
Acquisizione notturna con illuminatore, bg controllato e possibilità di segmentare efficacemente tanti dati reali row. La rete sarà allenata in condizioni notturne 

#Gestione dei dati di profondità.
Una volta ricavata una maschera di pixel (in lab senza bg, o in vigna con DNN), si estrarranno i dati di profondità di quei pixel e si…:
medieranno (n pixel ad una distanza media di d)
si convertirà ogni pixel già in mm^2 in base alla sua distanza dalla camera e si calcolera subito il volume reale 
Oppure senza segmentazione si valuterà la distanza dal filare in vigna con (RANSAC/ARUCO[tecnica depth o RGB])



## Getting Started

### Dependencies

* Librealsense is the foundamental library for acquisition, to acquire also the T265 camera realsese viewer is needed to fullfill all the requisites of the old T265 firmware
* Nvidia Jetpack, python 3.9, Opencv >4.6

### Installing

* We can provide you a requirementes.txt file if needed
* matlab script are customized to fit my laptop, other script are relative to the working directory, automatically filling the absolute path with your WD.

```
pip install pyrealsense2
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
```

### Executing program

* How to run the program
* Step-by-step bullets
```
python3.9 media_mapper_evaluator.py
python3.9 main.py
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@BernardoLanza]([https://www.linkedin.com/in/bernardo-lanza-554064163/])

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
