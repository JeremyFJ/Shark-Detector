# Shark Detector
Suitable shark conservation depends on well-informed population assessments. Direct methods such as scientific surveys and fisheries monitoring are adequate for defining population statuses, but species-specific indices of abundance and distribution coming from these sources are rare for most shark species. We can rapidly fill these information gaps by boosting media-based remote monitoring efforts with machine learning and automation.

We created a database of 53,345 shark images covering 219 species of sharks, and packaged object-detection and image classification models into a Shark Detector (SD) bundle. The SD recognizes and classifies sharks from videos and images using transfer learning and convolutional neural networks (CNNs). Our shark detection and classification pipeline is composed of several steps and three main components: 1 - an object-detection model called the Shark Locator (SL), which locates one or several shark subjects in images and draws bounding boxes around them; 2 - a binary sorting model called Shark Identifier (SI) which sorts images of sharks from a pool of heterogeneous images; and 3 - multiclass models called Shark Classifiers (SCs) which classify shark images to the genus and species levels.

We applied the SD to common data-generation approaches of sharks: collecting occurrence records from photographs taken by the public or citizen scientists, processing baited remote camera footage and online videos, and data-mining Instagram. We examined the accuracy of each model and tested genus and species prediction correctness as a result of training data quantity. The Shark Detector can classify 47 species pertaining to 26 genera. It sorted heterogeneous datasets of images sourced from Instagram with 91% accuracy and classified species with 70% accuracy. It located sharks in baited remote footage and YouTube videos with 89% accuracy, and classified located subjects to the species level with 69% accuracy.

<p align="center">
  <img src="pipeline.PNG" alt="pipeline" width="600"/>
</p>

## Models and Data
You can download the saved model weights [here.](https://drive.google.com/drive/folders/1KdVkSn4avPCa4iGjLp6Lf8IVSEAURQqs?usp=sharing)

The dataset structure is shown below
```
    ├── dataset                   <- root
        ├── training_set          <- training set folders        
        |   ├── not_shark      <- image files
        |   ├── shark
        |  
        ├── test_set              <- validation set folders
        |   ├── not_shark      <- image files
        |   ├── shark
``` 
## Code
This model implements the Keras package with a Tensorflow backend entirely in Python.  

### Requires
- [Anaconda / Python 3.7.3](https://www.anaconda.com/products/individual)
- [Tensorflow 2.2.0](https://www.tensorflow.org/)
- [Keras 2.3.1](https://keras.io/)
- [sklearn 0.21.2](https://scikit-learn.org/stable/)
- [PIL 6.1.0](https://pillow.readthedocs.io/en/stable/)

## Run
To train the model, run:
```
python main_train.py
```
Note -- Keep the image directory names static (or make changes to `img_loader.py`)
- (1) This script will load images.
- (2) New images can be added to not_shark/ and shark/ folders to increase the training and testing dataset.
- (3) Once predictions are made on the test set, accuracy and training history is reported in addition to confusion matrix plots for each class.

To classify images(s), run:
```
python main_inference.py
```
Note -- Add the images you want predicted to `model_predictor/dataset/test_set/shark/` 
- (1) Training weights saved as `.h5` are loaded for predictions as well as preprocessed data saved as `.npy` files.
- (2) You can adjust the threshold however you want. It is currently set to 0.90
- (3) `class_prediction.csv` is produced and `move_file.py` can be run to identify shark and not-shark predictions.

## Results

![confusion_matrix](cm_norm_50.png)

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
