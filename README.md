# Shark Detector
Suitable shark conservation depends on well-informed population assessments. Direct methods such as scientific surveys and fisheries monitoring are adequate for defining population statuses, but species-specific indices of abundance and distribution coming from these sources are rare for most shark species. We can rapidly fill these information gaps by boosting media-based remote monitoring efforts with machine learning and automation.

We created a database of 53,345 shark images covering 219 species of sharks, and packaged object-detection and image classification models into a Shark Detector (SD) bundle. The SD recognizes and classifies sharks from videos and images using transfer learning and convolutional neural networks (CNNs). Our shark detection and classification pipeline is composed of several steps and three main components: 1 - an object-detection model called the Shark Locator (SL), which locates one or several shark subjects in images and draws bounding boxes around them; 2 - a binary sorting model called Shark Identifier (SI) which sorts images of sharks from a pool of heterogeneous images; and 3 - multiclass models called Shark Classifiers (SCs) which classify shark images to the genus and species levels.

We applied the SD to common data-generation approaches of sharks: collecting occurrence records from photographs taken by the public or citizen scientists, processing baited remote camera footage and online videos, and data-mining Instagram. We examined the accuracy of each model and tested genus and species prediction correctness as a result of training data quantity. The Shark Detector can classify 47 species pertaining to 26 genera. It sorted heterogeneous datasets of images sourced from Instagram with 91% accuracy and classified species with 70% accuracy. It located sharks in baited remote footage and YouTube videos with 89% accuracy, and classified located subjects to the species level with 69% accuracy.

<p align="center">
  <img src="pipeline.PNG" alt="pipeline" width="600"/>
</p>

## Models and Data

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

## Run

```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Input your video into `video/`
- change your video name in `video_SD.py`
- `vid = [your video name]`
- make sure you have enough free memory  
`python video_SD.py`

## Results

<p align="center">
  <img src="ODM_1.PNG" alt="SD mako" width="800"/>
</p>

## Check out
Based on a multi-classification model trained to identify vector parasites of Schistosomiasis
- [Schisto-parasite-classification](https://github.com/deleo-lab/schisto-parasite-classification)

Web application `sharkPulse` designed to crowdsource shark images from around the world for monitoring shark populations 
- [sharkPulse](http://sharkpulse.org/)

## References
- [1] Jenrette J, Liu ZY-C, Hastie T, Ferretti F. Data mining Instagram as a tool for tracking global shark populations (TBD). 2020.
- [2] Liu ZY-C, Chamberlin AJ, Shome P, Jones IJ, Riveau G, Jouanard N, Ndione, Sokolow SH, De Leo GA. Identification of snails and parasites of medical importance via convolutional neural network: an application for human schistosomiasis. PLoS neglected tropical diseases. 2019.
- [3] Schroeder J. InstaCrawlR. Crawl public Instagram data using R scripts without API access token. 2018. 
- [4] Zisserman A, Simonyan K. Very Deep Convolutional Networks for Large-Scale Image Recognition. Published in arXiv. 2014.
- [5] Alexander Graf and André Koch-Kramer. Instaloader. A tool to download pictures (or videos) along with their captions and other metadata from Instagram. 2016. Ver 4.5.5.

## Contact
- Data: jjeremy1@vt.edu
- Model: jjeremy1@vt.edu, zacycliu@stanford.edu
