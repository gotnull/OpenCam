# import the necessary packages
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import numpy as np
import cv2

# import the necessary packages
from threading import Thread
import sys

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
  from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
  from Queue import Queue

stream = cv2.VideoCapture(0)
fps = FPS().start()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] Starting video stream thread...")
fvs = FileVideoStream(0).start()
time.sleep(1.0)
 
# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
while fvs.more():
  # grab the frame from the threaded video file stream, resize
  # it, and convert it to grayscale (while still retaining 3
  # channels)
  frame = fvs.read()
  frame = imutils.resize(frame, width=450)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = np.dstack([frame, frame, frame])

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
#[INFO] elapsed time: 13.65
#INFO] approx. FPS: 31.86

# When everything is done, release the capture and destroy all the windows
cv2.destroyAllWindows()
fvs.stop()
