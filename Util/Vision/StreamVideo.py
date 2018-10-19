import numpy as np
import scipy.signal as signal 
import scipy.ndimage as ndimage
import cv2 

def normalize_image(img):
    ep = np.atleast_3d(1e-12*np.std(img, axis = 2))
    return (img - np.atleast_3d(img.mean(axis = 2))) / (np.atleast_3d(img.std(axis = 2, dtype = np.float)) + ep)


def stream():
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Vertical Edge Detection", 2)
    cv2.namedWindow("Horizontal Edge Detection",3)
    cv2.namedWindow("Gradient Orientation",4)     
    kern1 = np.atleast_3d(np.array([[1,-1],[1,-1]]))           
    kern2 = np.atleast_3d(np.array([[1,-1],[-1,-1]]))            
    scharr = np.atleast_3d(np.array([[ -3-3j, 0-10j,  +3 -3j],
    [-10+0j, 0+ 0j, +10 +0j],         
    [ -3+3j, 0+10j,  +3 +3j]])) # Gx + j*Gy
    while True:                                     
        _, frame = cap.read()
        #print 'Frame Shape Before', frame.shape, "Type: ", type(frame)
        grad1 = normalize_image(ndimage.convolve(frame, kern1))                                          
        grad2 = normalize_image(ndimage.convolve(frame, kern2))       
        grad3 = normalize_image(ndimage.convolve(frame, scharr))                                           
        
        cv2.imshow('RGB', frame.mean(axis = 2))                                 
        cv2.imshow('Vertical Edge Detection', grad1)               
        cv2.imshow('Horizontal Edge Detection', grad2)
        cv2.imshow('Gradient Orientation',grad3)    
        if cv2.waitKey(40) == ord('a'):             
            print 'Now Exiting!'            
            break  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream()
