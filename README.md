

# MAPPER_AGRI_MULTICAM

acquisition and mapping system for agricoltural infield analisys.
Programma per valutare l'indice di crescita della vite tramite sistema di visione.

## Description

Questa repository contiene sia il software di acquisizione che il software di analisi per delle misure ottiche in agricoltura. 
Il file main.py permette di eseguire un programma in grado di acquisire e salvare immagini provenienti da una camera INTEL Realsense D435, tramita una scheda NVIDIA Jetson Nano.
Il file media_mapper_evaluator e il relativo file evaluator_utils (contenente le funzioni innestatate) permetto l analisi dei media prodotti dal main, analisi che viene condotta su PC. Siamo quinid in grado di produrre misure geometriche e volumetriche a partire daalle acquisizioni e stimare la biomassa lignea presente all interno delle nostre acquisizioni.

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
