import cv2
import numpy as np

video = (255*(np.load('vid.npy'))).astype(np.uint8).transpose([0,2,3,1])

#Save to video:
out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
for frame in video:
    out.write(frame)

out.release()

