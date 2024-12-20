# import the opencv library
import cv2
import os

# define a video capture object
vid = cv2.VideoCapture(3)

haar_get = os.path.join("cascades", "data", "haarcascade_frontalface_default.xml")
while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	frame = frame[120:120+250,200:200+250, :]
	
	haar_cascade = cv2.CascadeClassifier(haar_get)
	# converting to greyscale  
	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = haar_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
	for (x,y,w,h) in faces:

		width_cords = x+w
		height_cords = y+h
		cv2.rectangle(frame, (x,y), (width_cords,height_cords), (255,0,0), 2)
	# Display the resulting frame
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
