# Shark Detector
<p align="center">
  <img src="ODM_1.PNG" alt="SD mako" width="800"/>
</p>
We created a database of 53,345 shark images covering 219 species of sharks. The Shark Detector (SD) is a package of object-detection and image classification algorithms designed to identify sharks from visual media and classify 47 species with 70% accuracy using transfer learning and convolutional neural networks (CNNs).
<p>
<p align="center">
  <img src="pipeline.PNG" alt="pipeline" width="600"/>
</p>

## Models and Data

The SD is composed of a locating object-detector and two image classifiers bundled into the Shark Classifier (SC) architecture. The Shark Identifier (SI) is more tailored for big data mining, and so will not be used for video detection.
1) Shark Locator (SL) -- object-detection
2) Genus-specific classifier (GSC) -- image classifier (parent node)
3) Species-specific classifier (SSCg) -- image classifier (child node of genus)

We developed the SC as a hierarchical classification framework for classifying the identified shark images taxonomically. We trained one genus-specific model and a series of local species-specific models - one for each genus. The SC ingests the filtered shark images and classifies them at the genus level with the genus-specific classifier (GSC). Then, depending on the genus, a species-specific classifier (SSCg) would predict the most likely species. For the GSC, we trained 36,722 images across 26 genus classes. We trained 18 SSCg models with 19,243 images. The SC and SI models are continusously trained as new images are ingested.

You can download the saved model weights [here.](https://drive.google.com/drive/folders/1KdVkSn4avPCa4iGjLp6Lf8IVSEAURQqs?usp=sharing)

The dataset structure of training the GSC and SSCg is shown below
```
    ├── dataset                           <- root
        ├── training_set   <- GSC structure <- training set folders        
        |   ├── Alopias                     <- image files
                ├──Alopias vulpinus           <- SSCg structure
                ├──Alopias species 
        |   ├── Carcharhinus
        |   ├── Carcharias
        |  
        ├── test_set              
        |   ├── Alopias      
                ├──Alopias vulpinus  
                ├──Alopias species
        |   ├── Carcharhinus
        |   ├── Carcharias
``` 
## Code
This model implements the Keras package with a Tensorflow backend entirely in Python.  

### Requires
- [Python 3.8.10](https://www.python.org/downloads/)
- [Tensorflow 2.9.1](https://www.tensorflow.org/)
- [Keras 2.9.0](https://keras.io/)
- [PIL 9.1.1](https://pillow.readthedocs.io/en/stable/)
- [OpenCV 4.5.5.64](https://github.com/skvark/opencv-python)
- [pandas 1.2.3](https://pandas.pydata.org)
- [numpy 1.22.3](https://www.numpy.org)

The SD works best with GPU acceleration 
- CUDA version 11.0 
- NVIDIA driver 450.51.05

# Installation
Follow these instructions to install the Shark Detector package: 
### Mac / Linux
1. CLONE the repository and create your Python environment 
```
git clone https://github.com/JeremyFJ/Shark-Detector.git
cd Shark-Detector
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. DOWNLOAD a zipped folder **models** from [models](https://drive.google.com/drive/folders/1KdVkSn4avPCa4iGjLp6Lf8IVSEAURQqs?usp=sharing) (~3GB)
3. RUN:
```
cd ~/Downloads
mkdir -p models && tar -xvf models.tar.gz -C models
```
4. MOVE your **models/** directory to your cloned repository **Shark-Detector/**

### Windows


## Run
This repository currently instructs on how to detect and classify shark species from MP4 videos

See [sharkPulse](http://sharkpulse.cnre.vt.edu/can-you-find-a-shark/) to classify single images

- Move your video to `Shark-Detector/media/video/`
- Navigate to your repository directory `Shark-Detector/`
- Process one video:
`python video_SD.py [video name]`
- Process all videos in folder:
`python video_SD.py ALL`

## Results
`spreadsheets/[video name].csv` shows a csv file of all frames extracted and sharks classified
`frames/` outputed shark detected frames that are listed in the spreadsheet. Frames are labeled with the video name and the amount of seconds passed 

## Check out
- [Schisto-parasite-classification](https://github.com/deleo-lab/schisto-parasite-classification) Based on a multi-classification model trained to identify vector parasites of Schistosomiasis
- [Validation Monitor](http://sharkpulse.cnre.vt.edu/can-you-recognize/) Crowdsource shark images from around the world and involve citizen scientists to validate sightings
- [SeaQL](http://35.245.242.176/seaql/) Research Lab

## Contact
- Data: jjeremy1@vt.edu
- Model: jjeremy1@vt.edu, zacycliu@stanford.edu, pchimote@vt.edu
