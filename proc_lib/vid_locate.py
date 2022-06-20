import cv2

vid_dir = "video/"

def open_vid(vid):
    try:
        vidcap = cv2.VideoCapture(vid_dir+vid)
    except FileNotFoundError:
        print('video not found')
        sys.exit(2)
    return vidcap

def retrieve_frame(vidcap, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
    success,img = vidcap.read() 
    return success,img

def write_frame(frame, detection):
    cv2.imwrite(frame, cv2.resize(detection, (1024,1024))) # save frame as JPG file