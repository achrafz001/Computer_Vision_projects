import cv2
import matplotlib.pyplot as plt

#OpenCV has a of pre-trained classifiers for face detection, eye detection etc
cascade_classifier = cv2.CascadeClassifier('opencv_haarcascade_frontalface_alt.xml')

#opencv deals with BGR but Matplotlib deals with RGB
image = cv2.imread('girl.jpg')

#convert tp grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor =1.1, minNeighbors =10, minSize=(30,30))

for (x, y, width, height) in detected_faces :
    cv2.rectangle(image, (x,y), (x+width, y+width), (0,0,255),10)

#opencv uses BGR and Matpltlib uses RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
