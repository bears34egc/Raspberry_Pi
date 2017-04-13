import io
import picamera
import cv2
import numpy
from datetime import datetime
from pushbullet import Pushbullet

pb = Pushbullet('o.EZiuid3oCjU9aUM153BRs41bHYglKNee')

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
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

#Convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Look for faces in the image using the loaded cascade file
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

print "Found "+str(len(faces))+" face(s)"

#Draw a rectangle around every found face
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

#Save the result image
gt=datetime.now().strftime('%Y-%m-%d- %H:%M:%S - ')
cv2.imwrite('pic' +str(gt) + '.jpg',image)

with open('/home/ecoker/pic' + str(gt)+'.jpg', "rb") as pic:
    file_data = pb.upload_file(pic, str(gt)+'.jpg')
push = pb.push_file(**file_data)

push = pb.push_note("Cam is up and...","Up and Running")

