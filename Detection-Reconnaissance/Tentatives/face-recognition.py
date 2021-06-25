#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:05:31 2021

@author: karenhubert
"""


import face_recognition
import cv2
import numpy as np


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
alan_image = face_recognition.load_image_file("./training-data/s1/2.png")
alan_face_encoding = face_recognition.face_encodings(alan_image)[0]

JL_image = face_recognition.load_image_file("./training-data/s2/12.png")
JL_face_encoding = face_recognition.face_encodings(JL_image)[0]

joachim_image = face_recognition.load_image_file("./training-data/s3/12.png")
joachim_face_encoding = face_recognition.face_encodings(joachim_image)[0]

karen_image = face_recognition.load_image_file("./training-data/s4/8.png")
karen_face_encoding = face_recognition.face_encodings(karen_image)[0]

mohammed_image = face_recognition.load_image_file("./training-data/s5/26.png")
mohammed_face_encoding = face_recognition.face_encodings(mohammed_image)[0]

salvatore_image = face_recognition.load_image_file("./training-data/s6/1.png")
salvatore_face_encoding = face_recognition.face_encodings(salvatore_image)[0]

antoine_image = face_recognition.load_image_file("./training-data/s7/4.png")
antoine_face_encoding = face_recognition.face_encodings(antoine_image)[0]



known_face_encodings = [
    alan_face_encoding,
    JL_face_encoding,
    joachim_face_encoding,
    karen_face_encoding,
    mohammed_face_encoding,
    salvatore_face_encoding, 
    antoine_face_encoding
]

known_face_names = [
    "Alan ",
    "Jean-Luc", 
    "Joachim", 
    "Karen",
    "Mohammed",
    "Salvatore", 
    "Antoine"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()