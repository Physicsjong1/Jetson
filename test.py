from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import Jetson.GPIO as GPIO
from jetbot import Robot
robot = Robot()
for a in range(10):
    robot.forward(0.02*a)
    time.sleep(0.2)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(31,GPIO.OUT)

control=[]
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
signal=[]

# w     : width
# xm    : mean value of xmin and xmax
def human(w, xm, detections):
    det_hum = False
    for detection in detections:
        if detection[0].decode() == "person" :
            det_hum = True      # Human is detected in an image
            print("Human")

    #Classification of Distnace
    if(w > 100 and det_hum):
        print("Very close")
        control.append(2)
    elif( w > 60 and w <= 100 and det_hum):
        print("Close")
        control.append(1)
    else:
        print("Far")
        control.append(0)
    
    #Set Intervals
    if len(control) == 10:
        control.remove(control[0])
    if len(signal) == 2:
        signal.remove(signal[0])
    
    #Stop Definition
    if control.count(2) > 2:
        signal.append(2)
        if signal[0]==0: # Go
            if xm > 240:
                robot.set_motors(0.1,0.2)
                time.sleep(0.3)
                print("Stop and left turn")
                for aa in range(10):
                    robot.forward(0.2-aa*0.02)
                    time.sleep(0.2)
            else:
                robot.set_motors(0.2,0.1)
                time.sleep(0.3)
                print("Stop and right turn")
                for ab in range(10):
                    robot.forward(0.2-ab*0.02)
                    time.sleep(0.2)
        elif signal[0]==1: # Caution
            if xm > 240:
                robot.set_motors(0.1,0.2)
                time.sleep(0.3)
                print("Stop and left turn")
                for aa in range(10):
                    robot.forward(0.2-aa*0.02)
                    time.sleep(0.2)
            else:
                robot.set_motors(0.2,0.1)
                time.sleep(0.3)
                print("Stop and right turn")
                for ab in range(10):
                    robot.forward(0.2-ab*0.02)
                    time.sleep(0.2)
        robot.stop()
        time.sleep(1)
        for i in range(4):
            GPIO.output(31, GPIO.HIGH)
            time.sleep(0.125)
            GPIO.output(31, GPIO.LOW)
            time.sleep(0.125)

    # Caution Definition
    elif control.count(1) > 4:
        signal.append(1)
        print("Caution")
        if signal[0]==2: # Stop
            for b in range(10):
                robot.forward(b*0.02)
                time.sleep(0.2)
        robot.forward(speed=0.2)
        GPIO.output(31, GPIO.HIGH)
    
    # Go Definition
    else:
        signal.append(0)
        print("Go")
        if signal[0]==2: # Stop
            for b in range (10):
                robot.forward(0.02*b)
                time.sleep(0.2)
        GPIO.output(31, GPIO.LOW)
        robot.forward(0.2)

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        xm = (xmin + xmax) / 2
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        human(w, xm, detections)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2) 
    return img

netMain = None
metaMain = None
altNames = None

def YOLO():
    start=time.time()
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./bin/yolov3-tiny.weights"
    #configPath = "./cfg/yolov3.cfg"
    #weightPath = "./bin/yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    # JETBOT camera
    # cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, heigh=(int)720, format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
   # USB camera
    cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, width=640, height=360, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=360,format=BGR ! appsink")
   # cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#cap = cv2.VideoCapture("/home/nvidia-tx2/OpenCV_in_Ubuntu/Data/Lane_Detection_Videos/solidWhiteRight.mp4")
    #cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

    #cap.set(3, 1280)
    #cap.set(4, 720)
    #out = cv2.VideoWriter(
    #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        detections = darknet.detect_image(netMain, metaMain, frame_resized, thresh=0.25)
        if not detections:
            print("No objects")
            control.append(0)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(time.time()-prev_time)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        if(time.time()-start > 500):
            robot.stop()
            quit()
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
