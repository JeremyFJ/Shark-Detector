import numpy as np
import os
import tensorflow as tf
import cv2 as cv2
import shutil
import pickle
# from keras.utils import img_to_array, load_img
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import visualization_utils as vis_util
from PIL import Image

# Load a (frozen) Tensorflow model into memory.
PATH_TO_SAVED_MODEL="SL_modelv3"
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
specmod = dict() # load empty species model dictionary
spec_models = "/home/spr/SDv4/species/models/"
spec_labels = "/home/spr/SDv4/species/labels/"
gsc_labels = sorted(pickle.loads(open("/home/spr/SDv4/GSC.pickle", "rb").read()))
# Loading label map
shark_class_id = 1
category_index = {shark_class_id: {'id': shark_class_id, 'name': 'shark'}}

# Load CNN models
# optimizer = tfa.optimizers.RectifiedAdam()
def load_GSC():
    # model = load_model("/home/csteam/SDv3/GSC_VIT_mod/", compile=False)
    # model = load_model("/home/csteam/SDv3/model_gen.hdf5", compile=False)
    model = load_model("/home/spr/SDv4/GSC_mod/")
    print("GSC loaded")
    return model

def load_SC(mod, specmod):
    if mod not in specmod:
        # load class labels for each SSCg model
        classes = sorted(pickle.loads(open(spec_labels + mod + ".pickle", "rb").read()))
        # create dictionary of models and labels
        specmod[mod] = [load_model(spec_models + mod + "_mod"),classes] 
        print("\r" + mod + "--loaded")
        print("SSCg loaded\n")
    return specmod

def img_to_classify(det_img, size, batch_size):
    img_1 = img_to_array(det_img.resize(size))
    img_1 = img_1.reshape((batch_size, img_1.shape[0], 
                        img_1.shape[1], img_1.shape[2]))
    img_1 = img_1/255
    return img_1

def SC_predict(mod, img_1, labels):
    mod_pred = mod.predict(img_1)
    mod_pred_dict = dict(zip(labels, mod_pred.tolist()[0]))
    mod_top = sorted(mod_pred_dict, key=mod_pred_dict.get, reverse=True)[:1][0]
    print(mod_top)
    return mod_top

def detect(image_np, threshold):
    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections=detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
            for key,value in detections.items()}
    scores = detections['detection_scores']
    conf = scores[0]
    random_image = "none"
    # Each box represents a part of the image where a particular object was detected.
    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.int64)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=threshold,
        line_thickness=10,
        agnostic_mode=False)
    if conf > threshold:
        ymin = boxes[0,0]
        xmin = boxes[0,1]
        ymax = boxes[0,2]
        xmax = boxes[0,3]
        (im_height, im_width, channel) = image_np.shape
        (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn), int(xmaxx - xminn))
        img = cropped_image.numpy()
        random_image = Image.fromarray(img)

    return image_np, conf, random_image