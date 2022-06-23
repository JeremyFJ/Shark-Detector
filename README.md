# Shark Detector
Our shark detection and classification pipeline is composed of several steps and three main components: 1 - an object-detection model called the Shark Locator (SL), which locates one or several shark subjects in images and draws bounding boxes around them; 2 - a binary sorting model called Shark Identifier (SI) which sorts images of sharks from a pool of heterogeneous images; and 3 - multiclass models called Shark Classifiers (SCs) which classify shark images to the genus and species levels.

We applied the Shark Detector to common data-generation approaches of sharks: collecting occurrence records from photographs taken by the public or citizen scientists, processing baited remote camera footage and online videos, and data-mining Instagram. We examined the accuracy of each model and tested genus and species prediction correctness as a result of training data quantity. The Shark Detector can classify 47 species pertaining to 26 genera. It sorted heterogeneous datasets of images sourced from Instagram with 91% accuracy and classified species with 70% accuracy. It located sharks in baited remote footage and YouTube videos with 89% accuracy, and classified located subjects to the species level with 69% accuracy.

<img src="pipeline.PNG" alt="pipeline" class="center" width="600"/>

## Introduction
Suitable shark conservation depends on well-informed population statuses. While there are adequate direct methods to retrieve such data from scientific surveys and fisheries monitoring, species-specific indices of population abundance coming from these sources are rare for most shark species. These information gaps can be filled with image-based biomonitoring and machine learning. To encourage image-based monitoring and alleviate manual validation, we trained a shark image classification model using convolutional neural networks with a VGG-16 architecture to learn shark features and automatically discriminate shark images. Training and testing images were sourced from social networks and virtual archives. We collected over 24,000 images spanning 224 shark species. The trained model classified 2,500 images per minute on a desktop computer with a graphics processing unit (GPU) and 64 gigabytes of RAM. The model achieved 91% accuracy. We provided a GitHub repository which allows the user to access the training dataset and use the model. As image and video analyses strive to dominate methods for observing sharks in nature, an automated classifier can drastically reduce the burden of manually identifying datasets. Furthermore, a general shark classification and detection approach is a strong foundation for a species-specific framework.

In conjuncture with Instagram web scraping utilizing InstaCrawlR [3] and Instaloader [5] for post collection, this model can be used in a pipeline which maps inferred shark sightings according to locations mentioned in the post [1]. You can find this repository [here.](https://github.com/JeremyFJ/Instagram_sharkSighting)

![image_sample1.png](image_sample1.PNG)

Image examples for the two classes: not-shark, and shark

## Data
You can download the image repository and saved model weights [here.](https://drive.google.com/drive/folders/1B3zvSgJWfWQmo6mFgJZa4A1uoXSc0lUm?usp=sharing](https://drive.google.com/drive/folders/1KdVkSn4avPCa4iGjLp6Lf8IVSEAURQqs)

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
This model implements the Keras package with a Tensorflow backend entirely in Python. The pre-trained VGG16 is a CNN model built by Zisserman and Simonyan from the University of Oxford that achieves 92.7% accuracy from ImageNet with 14 million images and 1000 classes [4]. The final four layers are fully connected and explicitly written here after specifying `include_top=False`. The final layers are also frozen to better train the model to identify shark features. 

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
