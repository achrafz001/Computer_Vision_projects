import cv2

cascade_classifier = cv2.CascadeClassifier('opencv_haarcascade_frontalface_alt.xml')
#we will use real-time (camera) -0 meansopen the default camera
video = cv2.VideoCapture(0)

#setting the width and height of the video window
video.set(3,640)
video.set(4,480)

while True :
    #returns the next frame
    ret, img =video.read()
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    for (x, y, width, height) in detected_faces:
        cv2.rectangle(img, (x, y), (x + width, y + width), (0, 0, 255), 10)

        cv2.imshow('Real time face detection', img)
        # we wait for a key to be pressed- press 'ESC to quit
    key = cv2.waitKey(30) & 0xff
    if key == 27 :
        break

video.release()
cv2.destroyAllWindows()

