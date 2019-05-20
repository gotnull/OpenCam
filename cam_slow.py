from imutils.video import FPS
import imutils
import numpy as np
import cv2

stream = cv2.VideoCapture(0)
fps = FPS().start()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 320, 240)

while(True):
    # grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # draw a rectangle/bounding box around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # show the frame and update the FPS counter
    fps.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop the timer and display the FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#[INFO] elapsed time: 7.54
#INFO] approx. FPS: 5.70

# When everything is done, release the capture and destroy all the windows
stream.release()
cv2.destroyAllWindows()
