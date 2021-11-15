import cv2.cv2
import numpy as np
import cv2

original_image =cv2.imread('sunflower.jpg', cv2.IMREAD_COLOR)

# we have to transform the image into grayscale
# OpenCV handles BGR instead of RGB
gray_image =cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# LaplacianKERNEL
result_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('original image', original_image)
cv2.imshow('gray image', gray_image)
cv2.imshow('result image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()