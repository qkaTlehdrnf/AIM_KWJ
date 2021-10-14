import cv2
import numpy as np
import os
import time
print(cv2.__version__)

cap = cv2.VideoCapture("video.avi")
fps = 60
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    time.sleep(1/fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()