import sys
import numpy as np
import cv2
from naoqi import ALProxy
import face_recognition
import qi

Session = qi.Session()

Session.connect("tcp://192.168.2.147:9559")

tts = Session.service("ALTextToSpeech")

#if(len(sys.argv) <= 2):
#    print "parameter error"
#    print "python " + sys.argv[0] + " <ipaddr> <port>"
#    sys.exit()

#ip_addr = sys.argv[1]
#port_num = int(sys.argv[2])
ip_addr = "192.168.2.147"
port_num = 9559



# get NAOqi module proxy
videoDevice = ALProxy('ALVideoDevice', ip_addr, port_num)

# subscribe top camera
AL_kTopCamera = 0
AL_kQVGA = 1            # 320x240
AL_kBGRColorSpace = 13
captureDevice = videoDevice.subscribeCamera(
    "test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

# create image
width = 320
height = 240
image = np.zeros((height, width, 3), np.uint8)
process_this_frame = True

alan_image = face_recognition.load_image_file("./Detection/training-data/s1/2.png")
alan_face_encoding = face_recognition.face_encodings(alan_image)[0]

JL_image = face_recognition.load_image_file("./Detection/training-data/s2/12.png")
JL_face_encoding = face_recognition.face_encodings(JL_image)[0]

joachim_image = face_recognition.load_image_file("./Detection/training-data/s3/12.png")
joachim_face_encoding = face_recognition.face_encodings(joachim_image)[0]

karen_image = face_recognition.load_image_file("./Detection/training-data/s4/8.png")
karen_face_encoding = face_recognition.face_encodings(karen_image)[0]

mohammed_image = face_recognition.load_image_file("./Detection/training-data/s5/26.png")
mohammed_face_encoding = face_recognition.face_encodings(mohammed_image)[0]

salvatore_image = face_recognition.load_image_file("./Detection/training-data/s6/1.png")
salvatore_face_encoding = face_recognition.face_encodings(salvatore_image)[0]

antoine_image = face_recognition.load_image_file("./Detection/training-data/s7/4.png")
antoine_face_encoding = face_recognition.face_encodings(antoine_image)[0]

seen_face=[]

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



while True:

    # get image
    result = videoDevice.getImageRemote(captureDevice);

    if result == None:
        print result
        print 'cannot capture.'
    elif result[6] == None:
        print 'no image data string.'
    else:

        # translate value to mat
        values = map(ord, list(result[6]))
        i = 0
        for y in range(0, height):
            for x in range(0, width):
                image.itemset((y, x, 0), values[i + 0])
                image.itemset((y, x, 1), values[i + 1])
                image.itemset((y, x, 2), values[i + 2])
                i += 3

        # show image
        rgb_small_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
	#print face_locations

        if process_this_frame:
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
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
          #  top *= 4
          #  right *= 4
          #  bottom *= 4
          #  left *= 4

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	    if name not in seen_face:
	    	tts.say("Hello "+name)
		seen_face.append(name)
	    elif name=="Unknown":
		tts.say("Hello stranger")

		
	

        cv2.imshow("pepper-top-camera-320x240", image)

    # exit by [ESC]
    ##if cv2.waitKey(33) == 27:
    if cv2.waitKey(33)==ord('a'):
        break
