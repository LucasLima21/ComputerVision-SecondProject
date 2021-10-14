import cv2
import matplotlib.pyplot as plt
import numpy as np

def cannyFilter(frame):  
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # deixei em escala de cinza para manter a dimensão igual ao 
    canny = cv2.Canny(grayScale, 100, 150)
    result = np.hstack([grayScale,canny])
    cv2.imshow("Gray Scale Image and Canny Filter", result)  



capture = cv2.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    # cv2.imshow('Frames WebCam', frame)
    cannyFilter(frame)
    k = cv2.waitKey(30) & 0xff
    if(k == 27):
        break


capture.release()
cv2.destroyAllWindows()
