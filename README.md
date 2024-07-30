# Shark Detector v4 
[Shark detection and classification with machine learning, Jenrette et al., 2022](http://seaql.org/wp-content/uploads/2022/06/SD.pdf)

## Models and Data

The SD is composed of an object-detection model, Shark Locator (SL), and multiple image classifiers packaged into the Shark Classifier (SC). The Shark Identifier (SI) is a binary classification scheme which is more tailored to big data mining.
1) Shark Locator (SL) -- object-detection
2) Genus-specific classifier (GSC) -- image classifier (parent node)
3) Species-specific classifier (SSCg) -- image classifier (child node of genus)

We developed the SC as a hierarchical framework for taxonomically classifying located shark images. We trained one genus-specific model and a series of local species-specific models - one for each genus. The SC ingests the filtered shark images and classifies them at the genus level with the genus-specific classifier (GSC). Then, depending on the genus, a species-specific classifier (SSCg) will predict the most likely species. For the GSC, we trained 45,101 images across 26 genus classes. We trained 18 SSCg models with 24,391 images. The SC and SI models are continusously trained as new images are ingested.

# Installation
Follow these instructions to install and run the Shark Detector application: 
### Mac / Linux
1. CLONE the repository and create your Python environment 
```
git clone https://github.com/JeremyFJ/Shark-Detector.git
cd Shark-Detector
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. DOWNLOAD the file **models** from [models](https://www.kaggle.com/datasets/jeremyjer/sharkdetector) (~5GB)
3. EXTRACT and MOVE your **models** to your cloned repository
4. Run the SD on a test video 
```
python sd_bruv.py
```
5. Detection frames to `detvid/` and spreadsheet to `data/`

## Contact
- Data: jjeremy1@vt.edu
