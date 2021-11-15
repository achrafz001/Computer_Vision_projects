import numpy as np
import cv2

original_image =cv2.imread('sunflower.jpg', cv2.IMREAD_COLOR)

#sharpen kernel
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpen_image = cv2.filter2D(original_image,-1 , kernel)

# face recognition and we have a blurry CCTV video then we can apply this kernel
# in order to increase the precision of the underlying model

cv2.imshow('original image', original_image)
cv2.imshow('sharpen image', sharpen_image )
cv2.waitKey(0)
cv2.destroyAllWindows()