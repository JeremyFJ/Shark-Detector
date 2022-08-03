#########################################
############ Shark Detector #############
#########################################
# Code by: Jeremy Jenrette
# email: jjeremy1@vt.edu
# GitHub: JeremyFJ 
#-------------------------------------------------------------------------------------------------------
# Load libraries
#-------------------------------------------------------------------------------------------------------
from proc_lib.var import *
from proc_lib.detect import *
from proc_lib.mods import *
from proc_lib.classify import *
from proc_lib import vid_locate
from PIL import Image
import sys
import shutil
#-------------------------------------------------------------------------------------------------------
# Locate video
#-------------------------------------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove tensorflow warnings 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
vid_dir = "media/video/" # add video here
def usage():
    print("To run the Shark Detector, add your video(s) to the 'media/video/' directory")
    print("Process 1 video: python video_SD.py bruv1.mp4")
    print("Process all videos: python video_SD.py ALL\n")
try:
    os.remove("media/video/.gitignore")
    os.remove("media/processed_video/.gitignore")
except FileNotFoundError:
    pass
    
if __name__ == "__main__":
    if sys.argv[1].lower() == "all":
        video = os.listdir(vid_dir)
        if len(video)==0:
            print("\nThere are no videos in the 'media/video/' directory. Please add your videos and run again.")
            sys.exit(2)
    else:
        video = [sys.argv[1]]
        if sys.argv[1] not in os.listdir(vid_dir):
            print("\n" + str(video[0:]) + " cannot be found")
            usage()
            sys.exit(2)
#-------------------------------------------------------------------------------------------------------
# Load models: SL, SC: GSC, SSCg
#-------------------------------------------------------------------------------------------------------
print("\nprocessing " + str(video[0:]) + '\n')
detection_graph = load_SL()
gsc = load_SC(taxonomy="genus")     # load_SC in mods.py
sscg = load_SC(taxonomy="species")  
#-------------------------------------------------------------------------------------------------------
# Run SD
#-------------------------------------------------------------------------------------------------------
for vid in video:   # is either a list of one element or multiple elements depending on if ALL was specified 
    success = True
    count = 1
    vidcap = vid_locate.open_vid(vid)
    print("processing " + vid)
    print("Extracting frames")
    try:
        while success:
            success,img = vid_locate.retrieve_frame(vidcap, count)
            image = Image.fromarray(img)
            print("frame "+ str(count))
            # use the image object to locate a shark
            image_np = load_image_into_numpy_array(image) # image.resize(locate_size)
            with detection_graph.as_default():
                with tf.compat.v1.Session(graph=detection_graph) as sess:
                    det_img = detect_objects_crop_video(image_np, sess, detection_graph,
                                thresholdLoc=0.9979)     # locates a shark above this threshold, draws a box
            if str(det_img) == "none":
                count = count + interval_frame
            else:
                try:
                    frame = frame_save_path + vid.split(".")[0] + "_frame%d.jpg" % count
                    vid_locate.write_frame(frame, det_img) # save frame as JPG image at 1024x1024 px   
                except:
                    break
                frame = frame.split("frames/")[1]
                det_img = Image.fromarray(det_img)
                print("shark detected")
                img_1 = img_to_classify(det_img, gen_size, batch_size)
                gen_top3 = gsc_predict(gsc, img_1, gsc_labels) # returns top three genus predictions
                if any(gen_top3[0] in s for s in single_spec): # if top guess is a genus with only one species, SSCg is not used
                    dat = single_spec_check(gen_top3, single_spec, frame, dat)
                    count = count + interval_frame
                    continue
                if gen_top3[0] == "other_genus": # if top guess is a data-poor genus, SSCg is not used 
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
#-------------------------------------------------------------------------------------------------------
# Output spreadsheet and label video(s) as 'processed'
#-------------------------------------------------------------------------------------------------------
    dat.to_csv("spreadsheets/"+vid.split(".")[0]+".csv", index=False)
    if vid in os.listdir("media/processed_video/"):
        new_vid = vid.split(".")[0] + "_01" + "." + vid.split(".")[1]
        os.rename(vid_dir + vid,vid_dir + new_vid)
        shutil.move(vid_dir + new_vid, "media/processed_video/" + new_vid)
    else:
        shutil.move(vid_dir + vid, "media/processed_video/" + vid)
