# Source : https://www.superdatascience.com/blogs/opencv-face-recognition

import cv2
import os
import numpy as np 

subjects= ["",'Alan', 'Jean-Luc','Joachim', 'Karen', 'Mohammed', 'Salvatore', "Antoine"]

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []

    #let's go through each directory and read images within it
    for dir_name in dirs:

        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        

        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)

            #display an image window to show the image 
            #cv2.imshow("Training on image...", image)
            #cv2.waitKey(100)

            #detect face
            face, rect = detect_face(image)
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
        

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
        
def train():
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")
    
    #print total faces and labels
    print("Total faces: ", len(faces))
    print("exemple d'une face", faces[0])
    print("Total labels: ", len(labels))
    print("exemple d'un label", labels[0])

    #create our LBPH face recognizer 
    #face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img, face_recognizer):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    print("type face - rect", type(face),type(rect))
    print("face", face)
    print("rect", rect)

    #predict the image using our face recognizer 
    label = face_recognizer.predict(face)
    print(label[0])
    print("pourcentage", label[1])
    #get name of respective label returned by face recognizer
    label_text = subjects[label[0]]
    print(label_text)

    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

def prediction_alan(face_recognizer):
    print("Predicting images...")

    #load test images
    test_img1 = cv2.imread("test-data/Alan.png")
    #cv2.imshow("image1", test_img1)


    #perform a prediction
    predicted_img1 = predict(test_img1, face_recognizer)
    print("Prediction complete")
    #display both images
    cv2.imshow("Prediction", predicted_img1)
    #cv2.imshow(subjects[2], predicted_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def train_reconnaissance():
     face_recognizer = train()
     return face_recognizer 

def prediction_perso(img_path, face_recognizer):
    print("Prediction personne dans script_prediction")
    img = cv2.imread(img_path)
    try : 
        prediction = predict(img, face_recognizer)
    except: 
       prediction =  "unknown"
    return prediction 


def prediction_personne(name, face_recognizer):
    print("Predicting images...")

    #load test images
    path_img = "test-data/" + name +".png"
    test_img1 = cv2.imread(path_img)
    #cv2.imshow("image1", test_img1)


    #perform a prediction
    predicted_img1 = predict(test_img1, face_recognizer)
    print("Prediction complete")
    #display both images
    cv2.imshow("Prediction", predicted_img1)
    #cv2.imshow(subjects[2], predicted_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def main():
    print("Main")
    face_recognizer = train()
    prediction_personne('Karen', face_recognizer)

main()
