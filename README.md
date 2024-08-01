# Shark Detector v4 
[Shark detection and classification with machine learning, Jenrette et al., 2022](http://seaql.org/wp-content/uploads/2022/06/SD.pdf)

## Models and Data
The Shark Detector is composed of an object-detection and image classification ensemble. For simple function and use, we developed the R package, [sharkDetectorR](https://github.com/sharkPulse/sharkDetectorR). For increased customization, retraining, and increased processing speed, this repository offers the scripts and models to the public. Version 4.0.0 can classify 69 species of sharks with an average accuracy of 82%.  

# Installation

## Prerequisites
- Python 3.7 or higher

## Setting Up the Environment
1. CLONE the repository and create your Python environment 
```
git clone https://github.com/sharkPulse/Shark-Detector.git
cd Shark-Detector
python -m venv shark_env
source .shark_env/bin/activate
pip install -r requirements.txt
```
2. DOWNLOAD the file **v4_hawaii.tar.gz** from [models](https://www.kaggle.com/datasets/jeremyjer/sharkdetector) (~4GB)
3. EXTRACT and MOVE your **models** to your cloned repository
4. Run the Shark Detector on a test video 
```
python process_video.py
```
5. Detection frames and annotations to `output/`

## Running the Scripts

### Process Video
1. When prompted, enter the path to the video file you wish to process.
2. The script will process the video and save the results to the `./output/<video_name>/` directory.
3. A CSV file with the results will also be saved to the `./output/` directory.

### Process Images
1. Place all images to be processed in the `./images/` directory.
2. The script will process all images in this directory and move the processed images to the `./output/images/` directory.
3. A CSV file with the results will be saved to the `./output/` directory with the name format `images_<current_time>_results.csv`.

## Additional Notes
- Ensure that the models and labels are in the correct paths as referenced in the scripts.
- If you encounter any issues with TensorFlow GPU setup, ensure your system has the correct CUDA and cuDNN versions installed. Refer to the [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu) for detailed instructions.

## Contact
- Data: jjeremy1@vt.edu
