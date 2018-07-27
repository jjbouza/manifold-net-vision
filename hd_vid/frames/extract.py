import cv2
import numpy as np


def getVideoArray(name):
    #Load video into numpy array.
    cap = cv2.VideoCapture(name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, val = cap.read()
        if ret:
            buf[fc] = val.astype('float32')/255.0
        fc += 1

    cap.release()
    return buf

np.save('vid.npy', getVideoArray('out.mp4').transpose(0,3,1,2))
