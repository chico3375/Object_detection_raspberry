
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
# from picamera2
from picamera2 import Picamera2
import sys
import os
from os import system
import io
from os.path import isfile, join

from multiprocessing import Process


# find user
users  = []
users.append(os.getlogin())
tilt  = False
#This is to pull the information about what each object is called
classNames = []
classFile = "/home/" + users[0] + "/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

#This is to pull the information about what each object should look like
configPath = "/home/"+ users[0] + "/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/"+ users[0] + "/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

#This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


#This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label   
def getObjects(img, thres, nms, draw=True, objects=[], tilt=False):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
#Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below     
    #print(classIds,bbox)
    
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
                #print(className)
                tilt = True
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
    
    return img,objectInfo,tilt
    

#Classe Looper, capture 
class Looper:
    def __init__(self, time_count):
        self.time_count = time_count

    def start(self):
        start = time.time()
        end = start + self.time_count

        lastUploaded = datetime.datetime.now()
        # start Pi camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
    
# Boucle principale
        while time.time() <= end:
            
            timestamp = datetime.datetime.now()
            # GET AN IMAGE from Pi camera
            img = picam2.capture_array("main")             
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            #Below provides a huge amount of controll. the 0.45 number is the threshold number, the 0.2 number is the nms number)
            result, objectInfo, tilt = getObjects(img,0.45,0.2, objects=['person'] )
            if (tilt == True):                   
                p = person(img)
                p.start()
                #*********************
            #print(objectInfo)
            cv2.imshow("Output",img)
            
            k = cv2.waitKey(200)
            
            tilt = False
            if k == 27:    # Esc key to stop
                # EXIT
            
                cv2.destroyAllWindows()
                break
   
 
            

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
            print("Personne !")
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

