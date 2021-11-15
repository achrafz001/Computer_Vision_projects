import cv2
import numpy as np
#print (cv2.__version__)
# grayscale image

image =cv2.imread('sunflower.jpg', cv2.IMREAD_COLOR)
#values close to 0 are darker
#values closer to 255 are brighther
print(image.shape)
print(image)
print('max :',np.amax(image))
print('min :',np.amin(image))

cv2.imshow('computer vision',image)
cv2.waitKey(0)
cv2.destroyAllWindows()