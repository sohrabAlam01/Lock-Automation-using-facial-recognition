# code to extract the dat file from bz2 file
'''
import bz2
import os

# Check if .dat exists, otherwise extract
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as f_in:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
            f_out.write(f_in.read())
    print("Extracted shape_predictor_68_face_landmarks.dat")

'''

from copyreg import pickle
import pickle
import face_recognition
import cv2
import os
import numpy as np
import cvzone
import dlib
from imutils import face_utils
from scipy.spatial import distance

#capturing the video from camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#setting the height and width of the cam
cap.set(3, 640)
cap.set(4, 480)

#reading the background image
imgBackground = cv2.imread('Resources/background.png')

#importing the mode images(from Resources/Modes) into a list
folderModePath = 'Resources/Modes'
#extracting directory names
modePathList = os.listdir(folderModePath)

#storing images of people in a list imgModeList
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#print(len(imgModeList))

#Load the encoding file

print("Loading Encoded File...")
file = open('EncodedFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")



# Load facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# EAR threshold and blink count
EAR_THRESHOLD = 0.23
blink_detected = False

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear




while True:
    success, img = cap.read()
    mirrored_frame = cv2.flip(img, 1)

    # Convert to grayscale for dlib
    gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours (optional)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(mirrored_frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(mirrored_frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            blink_detected = True
        else:
            blink_detected = False

   #Resizing and converting the images to RGB
    imgS = cv2.resize(mirrored_frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

   #finding the location of the current face on cam
    faceCurFrame = face_recognition.face_locations(imgS)
   #finding the encoding of the current located face only not the whole image of cam
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

   #setting the position of cam on the background image
    imgBackground[162:162+480, 55:55+640] = mirrored_frame
    #setting the position of mode image on the background image
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    #comparing the encoding of currect face on cam with the saved encoding in file one by one
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)
        #return the index with the min face distance in list faceDis
        matchIndex = np.argmin(faceDis)


        #creats rectangle around face using cvzone

        #getting the face location
        y1, x2, y2, x1 = faceLoc
        #we initially reduce the size of image by 1/4th so multiplying by 4 again to get the exact location of the face
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)


        if matches[matchIndex] and blink_detected:
            print("Known Face Detected")
            print("Id: ", studentIds[matchIndex])
            print("matchIndex", matchIndex)
        #     y1, x2, y2, x1 = faceLoc
        #     y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        #     bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
        #     imgBackground = cvzone.cornerRect(imgBackground, bbox, rt = 0)


    #cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        exit(0)