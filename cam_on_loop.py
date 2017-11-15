import io
import picamera
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import numpy
from datetime import datetime
from pushbullet import Pushbullet
from time import sleep
import time

while True:



	#Create a memory stream so photos doesn't need to be saved in a file
	stream = io.BytesIO()

	#Get the picture (low resolution, so it should be quite fast)
	#Here you can also specify other parameters (e.g.:rotate the image)
	with picamera.PiCamera() as camera:
	    camera.resolution = (450, 300)
	    camera.capture(stream, format='jpeg')

	#Convert the picture into a numpy array
	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	#Now creates an OpenCV image
	image = cv2.imdecode(buff, 1)

	#Load a cascade file for detecting faces
	# face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
	face_cascade = cv2.CascadeClassifier('/home/ecoker/cam/haarcascade_frontalface_default.xml')
	#Convert to grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	#Look for faces in the image using the loaded cascade file
	faces = face_cascade.detectMultiScale(gray, 1.25, 3)#,outputRejectLevels=True)
	# print "face weights, ", faces[2]

	if len(faces) > 0:
		print len(faces)
		print "Found "+str(len(faces))+" face(s)"
		#Draw a rectangle around every found face
		for (x,y,w,h) in faces:
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
		#Save the result image
		# gt=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		# cv2.imwrite('/home/ecoker/cam/face' +str(gt) + '.jpg',image)
	else:
		pass

	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--images", required=True, help="path to images directory")
	# args = vars(ap.parse_args())

	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths
	# for imagePath in paths.list_images(args["images"]):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		# image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.10)
	print 'body weights, ', weights

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still peoplej
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
	# print("[INFO] {}: {} original boxes, {} after suppression".format(
	# 	len(rects), len(pick)))
	bodies = (len(rects))
	# show the output images
	if (bodies > 0) or len(faces) > 0:
		print "%02d" % bodies, " bodies"
		#Save the result image
		gt=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		cv2.imwrite('/home/ecoker/cam/body' +str(gt) + '.jpg',image)

		with open('/home/ecoker/cam/body' + str(gt)+'.jpg', "rb") as pic:
		    file_data = pb.upload_file(pic, str(gt)+'.jpg')
		push = pb.push_file(**file_data)

		push = pb.push_note('Cam'," Found "+str(len(faces))+" faces, " + str(bodies) + " bodies")

	else:
		pass
	time.sleep(4)

