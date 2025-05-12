import pickle

import cv2
import os

import face_recognition

#importing student images
folderPath = 'Images'
#extracting directory names within the 'Image' folder
pathList = os.listdir(folderPath)
# print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #splitting the directory name in two id and .png and using only id part (at index 0)
    studentIds.append(os.path.splitext(path)[0])
   # print(path)
   # print(os.path.splitext(path)[0])

print(studentIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        #since opencv uses color BGR and face_recognition user RGB hence convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
# print(encodeListKnown)
print("Encoding Complete")
'''
following code is storing the data stored in encodeListKnownWithIds into a binary file called "EncodedFile.p" 
using the pickle module, so we can load it later without  regenerating the encodings.
'''
#saving encodings with studentId in a file using pickle
file = open("EncodedFile.p", 'wb')
#serializing (converting the python data into byte stream ) encodeListKnownWithIds in order to save it in a file and write it to the file
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")