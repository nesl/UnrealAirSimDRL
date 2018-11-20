import numpy 
import cv2


def disparity():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    cv2.namedWindow("Left View", 1)
    cv2.namedWindow("Right View", 2)
    cv2.namedWindow("Disparity", 3)

    while True: 
        _, frameL = cap1.read()
        _, frameR = cap2.read()
        frameL = cv2.cvtColor(frameL, cv2.CV_8UC1)
        frameR = cv2.cvtColor(frameR, cv2.CV_8UC1)
        
        stereo = cv2.StereoBM_create(blockSize=15)
        dispar = stereo.compute(frameL,frameR)
        cv2.imshow('Left View', frameL)
        cv2.imshow('Right View', frameR)
        cv2.imshow('Disparity', dispar)
        if cv2.waitKey(10) == 27:
            print("Goodbye!")
            break



if __name__ == "__main__":
    disparity()
