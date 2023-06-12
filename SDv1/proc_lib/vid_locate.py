#-------------------------------------------------------------------------------------------------------
# Load libraries
#-------------------------------------------------------------------------------------------------------
import cv2

vid_dir = "media/video/"
#-------------------------------------------------------------------------------------------------------
# Video manipulation functions
#-------------------------------------------------------------------------------------------------------

# locate user input video
def open_vid(vid):
    try:
        vidcap = cv2.VideoCapture(vid_dir+vid)
    except FileNotFoundError:
        print('video not found')
        sys.exit(2)
    return vidcap

# split video into frames
def retrieve_frame(vidcap, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # captures 1 frame per second
    success,img = vidcap.read() 
    return success,img

# write frames
def write_frame(frame, detection):
    cv2.imwrite(frame, cv2.resize(detection, (1024,1024))) # save frame as JPG file