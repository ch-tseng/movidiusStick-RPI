from mvnc import mvncapi as mvnc
#from libMovidius.models import ssdMobilenets
from libMovidius.models import tinyYOLO2
from libMovidius.device.lcd import ILI9341
import numpy as np
import cv2
import sys
import io
import time
import picamera

lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=0)
dim=(300,300)
rotate = 90

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.OpenDevice()

# --> SSD_Mobilenets
#graphPath = 'ssdMobileNets/graph'
#LABELS = ('background',
#          'aeroplane', 'bicycle', 'bird', 'boat',
#          'bottle', 'bus', 'car', 'cat', 'chair',
#          'cow', 'diningtable', 'dog', 'horse',
#          'motorbike', 'person', 'pottedplant',
#          'sheep', 'sofa', 'train', 'tvmonitor')
#dimSize = (240,320)
#
# --> TinyYOLO2
graphPath = 'tinyYOLO2/graph'
LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
          "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
          "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
dimSize = (447,447)


def videoCamera():
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        #camera.rotation = rotate
        camera.resolution = (1024, 768)
        #camera.start_preview()
        #time.sleep(2)
        camera.capture(stream, format='jpeg', resize=(447, 447))

        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        cv_image = cv2.imdecode(data, 1)
        # OpenCV returns an array with data in BGR order. If you want RGB instead
        # use the following...
        #image = cv_image[:, :, ::-1]
        #lcd.displayImg(image)

        return cv_image

#model = ssdMobilenets(device, graphPath, LABELS)
model = tinyYOLO2(device, graphPath, LABELS)

while True:
    imgCaptured = videoCamera()

    image = model.run(imgCaptured)
    if image is not None:
        #image = image[:, :, ::-1]
        lcd.displayImg(image)
