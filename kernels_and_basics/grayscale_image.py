import cv2
#print (cv2.__version__)
# grayscale image

image =cv2.imread('sunflower.jpg', cv2.IMREAD_GRAYSCALE)
#values close to 0 are darker
#values closer to 255 are brighther
print(image.shape)
print(image)


cv2.imshow('computer vision',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
