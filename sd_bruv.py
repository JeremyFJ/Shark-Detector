import numpy as np
import os
import tensorflow as tf
import cv2 as cv2
import shutil
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import visualization_utils as vis_util
from PIL import Image

try:
    shutil.rmtree("frames/neg/")
    shutil.rmtree("frames/pos/")
except (FileNotFoundError, FileExistsError):
    pass
os.makedirs("frames/neg")
os.makedirs("frames/pos")

# Load a (frozen) Tensorflow model into memory.
PATH_TO_SAVED_MODEL="SL_modelv3"
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
specmod = dict() # load empty species model dictionary
spec_models = "/home/spr/SDv4/species/models/"
spec_labels = "/home/spr/SDv4/species/labels/"
gsc_labels = sorted(pickle.loads(open("/home/spr/SDv3/GSC.pickle", "rb").read()))
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

def open_vid(videofile):
    try:
        vidcap = cv2.VideoCapture(vid_dir+videofile)
    except FileNotFoundError:
        print('video not found')
        vidcap='none'
    fps=vidcap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    return vidcap

def retrieve_frame(vidcap, count, frame_cap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*frame_cap)) 
    success,img = vidcap.read()
    return success,img

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
        # img = np.array(cropped_image)
        img = cropped_image.numpy()
        random_image = Image.fromarray(img)
        # random_image = img_to_array(random_image.resize((224,224)))
        # random_image = tf.expand_dims(random_image, 0)
        # img = img_to_array(cropped_image)
        # random_array = np.random.random_sample(img.shape) * 255
        # random_array = random_array.astype(np.uint8)
        # random_image = Image.fromarray(random_array)

    return image_np, conf, random_image

# Checks and deletes the output file
# You cant have a existing file or it will through an error
vid = 'greyreef.mp4'
vid_dir = 'www/video/'
FILE_OUTPUT = vid_dir + 'greyreef.avi'
if os.path.isfile(vid_dir + FILE_OUTPUT):
    os.remove(vid_dir + FILE_OUTPUT)
# Playing video from file
cap = open_vid(vid)
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M','J','P','G'),
                      30, (frame_width, frame_height))
ret = True
count = 1
interval_frame = 1
frame_cap = 30 # grabs 1 frame per second
# for large videos
# mb_size = (os.path.getsize(vid_dir+vid)) * 10 ** -6 
# if mb_size>250: # checks if video is larger than 250mb
#     frame_cap = 100

gsc = load_GSC() # load genus model

while ret:
    # Capture frame-by-frame
    ret, frame1 = retrieve_frame(cap, count, frame_cap)
    if ret == True:
        thresh = 0.80
        frame2, conf, cropped_image = detect(frame1, thresh)
        name = "frames/neg/frame%d.jpg"%count
        if conf>thresh:
            name = "frames/pos/frame%d.jpg"%count
            img_1 = img_to_classify(cropped_image, (224,224), 1)
            gen_top = SC_predict(gsc, img_1, gsc_labels)
            mod_top = gen_top
            if ((gen_top + "_mod") in os.listdir(spec_models)):
                specmod = load_SC(gen_top, specmod)
                mod_top = SC_predict(specmod[gen_top][0], img_1, specmod[gen_top][1])
                mod_top = ' '.join(mod_top.split("_"))
            # Saves for video
            frame2 = cv2.putText(frame2, mod_top, (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (57,255,20), 2, cv2.LINE_AA, False)
            cv2.imwrite(name, frame1)
            cv2.imwrite("live.jpg", retrieve_frame(cap, count, frame_cap)[1])
        else:
            # uncomment below if you want to save negative frames
            # cv2.imwrite(name, frame2)
            pass
        out.write(frame1)
        count = count + interval_frame
        print('frame ' + str(count) + ' saved')
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()