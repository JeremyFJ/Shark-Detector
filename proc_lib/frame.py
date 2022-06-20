# build function to extract frame

def vid_retrieve(vid):
    import cv2
    from PIL import Image
    from tensorflow.keras.preprocessing.image import img_to_array
    vid_dir = "video/"
    try:
        vidcap = cv2.VideoCapture(vid_dir+vid)
    except FileNotFoundError:
        print('video not found')
        sys.exit(2)
    success,img = vidcap.read()
    interval_frame = 1
    count = 1
    thresholdLoc = 0.9
    locate_size = (512,512)
    gen_size = (256,256)
    spec_size = (128,128)
    success = True
    frame_save_path = 'frames/' 
    print('\nExtracting frames')
    gsc.run_eagerly = True
    try:
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                while success:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
                    success,img = vidcap.read()
                    image = Image.fromarray(img)
                    # use the image object to locate a shark
                    image_np = load_image_into_numpy_array(image.resize(locate_size))
                    det_img = detect_objects_crop_video(image_np, sess, detection_graph)
                    if det_img == "none":
                        count = count + interval_frame
                        continue
                    img_1 = Image.fromarray(det_img).resize(gen_size)
                    x = img_to_array(img_1)
                    x = np.expand_dims(x, axis=0)
                    images = np.vstack([x])
                    gen_pred = gsc.predict(images, batch_size=10)
                    print('image located')
                    count = count + interval_frame
    except (UnboundLocalError,ValueError) as e:
        print(e)

# Load the Shark Locator 
import tensorflow as tf
PATH_TO_CKPT = "models/Locator_model.pb"
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("Shark Locator loaded")

def detect_objects_crop_video(image_np, sess, detection_graph, thresholdLoc=0.99):
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
        img_data = 'none'
    return img_data

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)