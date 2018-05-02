from mvnc import mvncapi as mvnc
from libMovidius.device.lcd import ILI9341
import numpy as np
import cv2
import sys
import io
import time
import picamera

dpModel = "Agenet"   #tinyYolo2, sddMobileNets, Agenet
lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=0)
rotate = 90

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.OpenDevice()

if(dpModel=="sddMobileNets"):
    from libMovidius.models import ssdMobilenets
    graphPath = 'ssdMobileNets/graph'
    LABELS = ('background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
    dimSize = (240,320)
    model = ssdMobilenets(device, graphPath, LABELS)

elif(dpModel=="tinyYolo2"):
    from libMovidius.models import tinyYOLO2
    graphPath = 'tinyYOLO2/graph'
    LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
              "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
              "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    dimSize = (448,448)

    model = tinyYOLO2(device, graphPath, LABELS)

elif(dpModel=="Agenet"):
    from libMovidius.models import Agenet
    graphPath = "Agenet/graph"
    meanPath = "Agenet/age_gender_mean.npy"
    age_list=['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
    dimSize = (227,227)
    model = Agenet(mvnc, device, graphPath, meanPath, dimSize, age_list)

def videoCamera():
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        #camera.rotation = rotate
        camera.resolution = (1024, 768)
        #camera.start_preview()
        #time.sleep(2)
        camera.capture(stream, format='jpeg', resize=dimSize)

        # Construct a numpy array from the stream
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image from the array, preserving colour
        cv_image = cv2.imdecode(data, 1)
        # OpenCV returns an array with data in BGR order. If you want RGB instead
        # use the following...
        #image = cv_image[:, :, ::-1]
        #lcd.displayImg(image)

        return cv_image


while True:
    imgCaptured = videoCamera()

    image = model.run(imgCaptured, 0.60)
    if image is not None:
        #image = image[:, :, ::-1]
        lcd.displayImg(image)
