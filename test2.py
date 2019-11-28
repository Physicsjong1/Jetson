from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import Jetson.GPIO as GPIO
print("Starting Jetbot ......")
from jetbot import Robot

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

robot = Robot()

# w     : width
def human(w):
    if w > 120:
        control.append(2)
    elif w > 90 and w <= 120:
        control.append(1)
    else:
        control.append(0)

def inc1():
    for i in range(10):
        robot.forward(0.01*i)
        time.sleep(0.1)

def inc2():
    for i in range(10):
        robot.forward(0.1+0.01*i)
        time.sleep(0.1)

def inc3():
    for i in range(20):
        robot.forward(0.01*i)
        time.sleep(0.1)

def dec1():
    for i in range(10):
        robot.forward(0.1-0.01*i)
        time.sleep(0.1)
    robot.stop()

def dec2():
    for i in range(10):
        robot.forward(0.2-0.01*i)
        time.sleep(0.1)

def turn(xm):
    if xm > 240:
        robot.set_motors(0.1,0.2)
        time.sleep(1)
    else:
        robot.set_motors(0.2,0.1)
        time.sleep(1)

def signalcontrol(xm):
    if control.count(2) > 5:
        signal.append(2)
        print("STOP")
        GPIO.output(31, GPIO.HIGH)
        if signal[0] == 1:
            turn(xm)
            dec1()
        elif signal[0] == 0:
            dec2()
            turn(xm)
            dec1()
        robot.stop()
        for i in range(2):
            GPIO.output(31, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(31, GPIO.LOW)
            time.sleep(0.2)
    elif control.count(1) > 4:
        signal.append(1)
        GPIO.output(31, GPIO.HIGH)
        print("CAUTION")
        if signal[0] == 2:
            inc1()
        elif signal[0] == 0:
            dec2()
        robot.forward(0.1)
    else:
        signal.append(0)
        print("GO")
        if signal[0] == 2:
            inc3()
        elif signal[0] == 1:
            inc2()
        robot.forward(0.2)
        GPIO.output(31, GPIO.LOW)

def cvDrawBoxes(detections, img):
    xm = 0
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        if detection[0].decode() == "person":
            human(w)
            xm = (xmin + xmax) / 2
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2) 
    return img, xm

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
    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, heigh=(int)720, format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
   # cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#cap = cv2.VideoCapture("/home/nvidia-tx2/OpenCV_in_Ubuntu/Data/Lane_Detection_Videos/solidWhiteRight.mp4")
    #cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, width=640, height=360, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=360,format=BGR ! appsink")
    #cap.set(3, 1280)
    #cap.set(4, 720)
    #out = cv2.VideoWriter(
    #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3) 
    inc3()
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        while len(control) >= 10:
            control.remove(control[0])
        while len(signal) >= 2:
            signal.remove(signal[0])

        detections = darknet.detect_image(netMain, metaMain, frame_resized, thresh=0.25)
        det = False
        for d in detections:
            if d[0].decode() == "person":
                det = True
        print("####################\n")
        print(control)
        if not detections or not det:
            control.append(0)
            print("No Human")
        elif det:
            print("Human")

        image = cvDrawBoxes(detections, frame_resized)[0]
        xm = cvDrawBoxes(detections, frame_resized)[1]
        signalcontrol(xm)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Time: "+str(round(time.time()-prev_time,4)))
        print("\n####################")
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        if(time.time()-start > 500):
            robot.stop()
            quit()
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
