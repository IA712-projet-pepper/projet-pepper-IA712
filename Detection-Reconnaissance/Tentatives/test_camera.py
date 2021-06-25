#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:02:11 2021

@author: karenhubert
"""

"""

# importer les paquets nécessaires
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
# construire l'argument parser et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
				help="chemin d'accès à l'image d'entrée")
args = vars(ap.parse_args())

# initialiser le détecteur de visage de dlib (basé sur HOG)
detector = dlib.get_frontal_face_detector()
# répertoire de modèles pré-formés
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""

import cv2 as cv


dirCascadeFiles = r'../opencv/haarcascades_cuda/'
# Get files from openCV : https://github.com/opencv/opencv/tree/3.4/data/haarcascades
classCascadefacial = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
 
def facialDetectionAndMark(_image, _classCascade):
    imgreturn = _image.copy()
    gray = cv.cvtColor(imgreturn, cv.COLOR_BGR2GRAY)
    faces = _classCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv.rectangle(imgreturn, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return imgreturn

def videoDetection(_haarclass):
    webcam = cv.VideoCapture(0)
    if webcam.isOpened():
        while True:
            bImgReady, imageframe = webcam.read() # get frame per frame from the webcam
            if bImgReady:
                face = facialDetectionAndMark(imageframe, _haarclass)
                cv.imshow('My webcam', face) # show the frame
            else:
                print('No image available')
            keystroke = cv.waitKey(20) # Wait for Key press
            if (keystroke == 27):
                break # if key pressed is ESC then escape the loop
 
        webcam.release()
        cv.destroyAllWindows()   
         



print(cv.getBuildInformation())
webcam = cv.VideoCapture(0)
webcam.isOpened()


if webcam.isOpened():
    while True:
        bImgReady, imageframe = webcam.read() # get frame per frame from the webcam
        if bImgReady:
            cv.imshow('My webcam', imageframe) # show the frame
        else:
            print('No image available')
        keystroke = cv.waitKey(20) # Wait for Key press
        if (keystroke == 27):
            break # if key pressed is ESC then escape the loop
 
    webcam.release()
    cv.destroyAllWindows() 

"""

from time import perf_counter
t1_start = perf_counter()
frame_count = 0
webcam = cv.VideoCapture(0)
NB_IMAGES = 100
 
if webcam.isOpened():
    while (frame_count < NB_IMAGES):
        bImgReady, imageframe = webcam.read() # get frame per frame from the webcam
        frame_count += 1
         
    t1_stop = perf_counter()
    print ("Frame per Sec.: ", NB_IMAGES / (t1_stop - t1_start))
 
    webcam.release()
    cv.destroyAllWindows()
"""
    
    
videoDetection(classCascadefacial)
