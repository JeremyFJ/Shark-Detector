#--------------------------------------------------------------------------
# Load libraries
#--------------------------------------------------------------------------
from proc_lib.var import *
from proc_lib.detect import *
from proc_lib.mods import *
from proc_lib.classify import *
from proc_lib import vid_locate
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vid = "requiem_1.mp4"
print("processing " + vid + '\n')
detection_graph = load_SL()
gsc = load_SC(taxonomy="genus")
sscg = load_SC(taxonomy="species")

vidcap = vid_locate.open_vid(vid)
print("Extracting frames")
try:
    while success:
        success,img = vid_locate.retrieve_frame(vidcap, count)
        image = Image.fromarray(img)
        print("frame "+ str(count))
        # use the image object to locate a shark
        image_np = load_image_into_numpy_array(image) #image.resize(locate_size)
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                det_img = detect_objects_crop_video(image_np, sess, detection_graph,
                            thresholdLoc=0.9979)
        if str(det_img) == "none":
            count = count + interval_frame
        else:
            try:
                frame = frame_save_path + vid.split(".")[0] + "_frame%d.jpg" % count
                vid_locate.write_frame(frame, det_img) # save frame as JPG file
            except:
                break
            frame = frame.split("frames/")[1]
            det_img = Image.fromarray(det_img)
            print("shark detected")
            img_1 = img_to_classify(det_img, gen_size, batch_size)
            gen_top3 = gsc_predict(gsc, img_1, gsc_labels)
            if any(gen_top3[0] in s for s in single_spec):
                dat = single_spec_check(gen_top3, single_spec, frame, dat)
                count = count + interval_frame
                continue
            if gen_top3[0] == "other_genus":
                frame_df = pd.DataFrame([[frame, gen_top3[0], ""]], 
                        columns=list(dat.columns))
                dat = pd.concat([frame_df, dat])
                count = count + interval_frame
                continue
            img_1 = img_to_classify(det_img, spec_size, batch_size)
            dat = sscg_predict(sscg, img_1, frame, gen_top3, dat)
            count = count + interval_frame
except (UnboundLocalError,ValueError,AttributeError) as e:
    pass
dat.to_csv("spreadsheets/"+vid.split(".")[0]+".csv", index=False)