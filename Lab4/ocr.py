import cv2
import numpy as np
 
img = cv2.imread('download.jpeg')
 
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
noise = img + img * gauss
 
cv2.imshow('a',noise)
cv2.waitKey(0)