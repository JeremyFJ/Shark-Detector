#-------------------------------------------------------------------------------------------------------
# Load libraries
#-------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
#-------------------------------------------------------------------------------------------------------
# Object-detection functions
#-------------------------------------------------------------------------------------------------------

# Load the Shark Locator
def load_SL():
    # Load the Shark Locator 
    PATH_TO_CKPT = "models/Locator_model.pb"
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Shark Locator loaded")
    return detection_graph

# Locate shark from image, and draw a box for a new image
def detect_objects_crop_video(image_np, sess, detection_graph, thresholdLoc=0.9989):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # crop the bounding boxes
    if scores[0][0] >= thresholdLoc:
        ymin = boxes[0,0,0]
        xmin = boxes[0,0,1]
        ymax = boxes[0,0,2]
        xmax = boxes[0,0,3]
        (im_height, im_width, channel) = image_np.shape
        (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn), int(xmaxx - xminn))
        sess = tf.compat.v1.Session()
        img_data = sess.run(cropped_image)
        print('shark confidence: ' + str(scores[0][0]))
    else:
        img_data = str("none")
    return img_data

# format image
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)