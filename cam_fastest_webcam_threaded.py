# import the necessary packages
from __future__ import print_function
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
 
# loop over some frames...this time using the threaded stream
while(True):
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
  frame = vs.read()
  frame = imutils.resize(frame, width=400)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = np.dstack([frame, frame, frame])

	# check to see if the frame should be displayed to our screen
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

  cv2.imshow("Frame", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
 
  # update the FPS counter
  fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#[INFO] elapsed time: 7.92
#[INFO] approx. FPS: 41.56

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
