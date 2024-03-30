#for foscam
import urllib.request
from urllib.request import urlopen
#from PIL
from PIL import Image

import numpy as np
import imutils
import time
import datetime
import dropbox
import json
import cv2
from utils import string_to_image
from utils import image_to_string
from pyimagesearch.tempimage import TempImage
import argparse
import warnings
from freesms import FreeClient
# from NCS2
import sys
from os import system
import io
from os.path import isfile, join

from multiprocessing import Process

#Classe Looper, capture Foscam
class Looper:
    def __init__(self, time_count):
        self.time_count = time_count

    def start(self):
        start = time.time()
        end = start + self.time_count

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        lastUploaded = datetime.datetime.now()
        detectframecount = 0
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        out = None
# Boucle principale
        while time.time() <= end:
            label = ""
            #url = conf["url_cam"] + "/snapshot.cgi?user=" + conf["cam_login"] + "&pwd=" + conf["cam_password"]
            #url = "http://192.168.1.80:8000/stream.mjpg"
            url = "http://192.168.1.79:8081/snapshot.cgi?user=admin&pwd=foscam2bext"
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            image = response.read()
            image_as_file = io.BytesIO(image)
#décodage vers NumPy image
            frame = cv2.imdecode(np.fromstring(image_as_file.read(), np.uint8), 1)
#Visualisation des images numPy
#            cv2.imshow('foscam', image)
#            cv2.waitKey(1)
#Visualisation des images Pil
#            image_as_pil = Image.open(image_as_file)
#            image_as_pil.show()
#Sauvegarde des images
#        img_name = "Cam1-"  + "_" + self.loop_name + "_.jpg"
#        img_path = os.path.join(wdir, img_name)
#        image_as_pil.save(img_path)


            #frame = imutils.resize(frame, width=400)
            timestamp = datetime.datetime.now()
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.4:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                        confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    
            
                
                #test person detected
                if 'person' in label:
                    p = person(frame)
                    p.start()
                #*********************
            cv2.imshow('Camera portail', frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

#*********************************************************
class person:
    def __init__(self, f):
        self.f = f
    def start(self):
        """
        process dropbox
        """
        global lastUploaded
        image = self.f
        timestamp = datetime.datetime.now()
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
#            print("Personne !")
            # check to see if dropbox sohuld be used
            if conf["use_dropbox"]:
                ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
                # write the image to temporary file
                t = TempImage()
                cv2.imwrite(t.path, image)
                # upload the image to Dropbox and cleanup the tempory image
                print("[UPLOAD] {}".format(ts))
                path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
                client.files_upload(open(t.path, "rb").read(), path)
                t.cleanup()
                #Envoi SMS sur Free
                a = FreeClient(user=conf["free_access_token"], passwd=conf["free_password"]) # Ajoutez votre identifiant FreeMobile et votre clé API
                resp = a.send_sms("Alerte intrusion Misincu")
                resp.status_code  # 200
                # update the last uploaded timestamp 
            lastUploaded = timestamp

# *********************************************************
#MAIN   MAIN   MAIN
if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True,
        help="path to the JSON configuration file")
    args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
    warnings.filterwarnings("ignore")
    conf = json.load(open(args["conf"]))
    client = None
# check to see if the Dropbox should be used
    if conf["use_dropbox"]:
    # connect to dropbox and start the session authorization process
        client = dropbox.Dropbox(conf["dropbox_access_token"])
        print("[SUCCESS] dropbox account linked")
    lastUploaded = datetime.datetime.now()
#Lancement du process looper
    print('loop to be started')
    time_count = conf["count_time"]
    l = Looper(time_count)
    l.start()



