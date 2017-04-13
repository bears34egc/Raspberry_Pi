import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import tornado.options
import serial
import time
from datetime import timedelta
import cv2
import time
from datetime import datetime
from pushbullet import Pushbullet
import io
import numpy
import picamera

pb = Pushbullet('o.EZiuid3oCjU9aUM153BRs41bHYglKNee')

#for webcam users
camera=cv2.VideoCapture(0)

#for picam users
#import picam
#camera=picam.OpenCVCapture()


#if you prefer to change the resolution of the image otherwise comment below 2 lines
ret = camera.set(3,320) #width
ret = camera.set(4,240) #height

#ret=camera.set(10,0.6)

face_cascade = cv2.CascadeClassifier('/home/ecoker/.virtualenvs/cv/visitor_project/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

clients = []
f=open("/home/ecoker/.virtualenvs/cv/visitor_project/register.txt","a")





class WSHandler(tornado.websocket.WebSocketHandler):

  def check_origin(self, origin):
    return True


  def open(self):
    print 'A Client Is Connected'
    clients.append(self)


  def on_message(self, message):

    print 'Incoming status', message
    #a=message.split("!")





    if message=='who':
       count=0
       list1=[]
       a=""
       f=open("/home/ecoker/.virtualenvs/cv/visitor_project/register.txt","r")
       for line in f.readlines():

           if len(line) != 1 :


               list1.append(line)





           #count=count+1


       f.close()
       a=''.join(map(str,list1))
       self.write_message(a)



  def on_close(self):
    print 'Client Closed the Connection '
    clients.remove(self)

def send_message_to_clients(msg):
  for client in clients:
      client.write_message(msg)




def function_second():

   # ret, image=camera.read()

   #Create a memory stream so photos doesn't need to be saved in a file
   stream = io.BytesIO()

   #Get the picture (low resolution, so it should be quite fast)
   #Here you can also specify other parameters (e.g.:rotate the image)
   with picamera.PiCamera() as camera:
       camera.resolution = (450, 325)
       camera.capture(stream, format='jpeg')

#Convert the picture into a numpy array
   buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

   #Now creates an OpenCV image
   image = cv2.imdecode(buff, 1)

   print image.shape
  # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


  # faces = face_cascade.detectMultiScale(gray, 1.3, 4)

   # faces = face_cascade.detectMultiScale(gray,
			# 	scaleFactor=1.3,
			# 	minNeighbors=3,
			# 	minSize=(30,30),
			# 	flags=cv2.CASCADE_SCALE_IMAGE)

   #Look for faces in the image using the loaded cascade file
   faces = face_cascade.detectMultiScale(gray, 1.1, 5)
   print "Found "+str(len(faces))+" face(s)"

#Draw a rectangle around every found face
   for (x,y,w,h) in faces:
          cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)



   if len(faces)>=1:
          gt=datetime.now().strftime('%Y-%m-%d- %H:%M:%S - ')
          cv2.imwrite('/home/ecoker/.virtualenvs/cv/visitor_project/' + str(gt)+'.jpg',image)
          with open('/home/ecoker/.virtualenvs/cv/visitor_project/' + str(gt)+'.jpg', "rb") as pic:
            file_data = pb.upload_file(pic, str(gt)+'.jpg')
          push = pb.push_file(**file_data)
          # push - pb.push_link("We have " + str(len(faces))+" Visitors")
          push = pb.push_note("We have ...", str(str(len(faces)) + '__visitors'))
          m="log-"+gt+str(len(faces))+" Visitors"

          f.write("\n"+m)

   tornado.ioloop.IOLoop.instance().add_timeout(timedelta(seconds=1),
                                                     function_second)

if __name__ == "__main__":

  tornado.options.parse_command_line()
  application=tornado.web.Application(handlers=[

(r"/ws",WSHandler),


(r'/visitor_project/(.*)',tornado.web.StaticFileHandler,{'path':'/home/ecoker/.virtualenvs/cv/visitor_project'})

])

  http_server = tornado.httpserver.HTTPServer(application)
  http_server.listen(3030)
  tornado.ioloop.IOLoop.instance().add_timeout(timedelta(seconds=1),
                                                 function_second)
  tornado.ioloop.IOLoop.instance().start()
