import cv2
import face_recognition
import pickle
import os

path = 'dataset'
images = []
classNames = []
myList = os.listdir(path)

print(f'Initializing Feature Encoding for {len(myList)} images...')

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None: continue
    
    images.append(curImg)
    # This strips '_1' or '.jpg' so 'ayesha_1.jpg' becomes 'AYESHA'
    raw_name = os.path.splitext(cl)[0].split('_')[0]
    classNames.append(raw_name.upper())

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Using 'large' model for better feature extraction
        encodes = face_recognition.face_encodings(img, model="large")
        if len(encodes) > 0:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)
with open("EncodeFile.p", "wb") as f:
    pickle.dump([encodeListKnown, classNames], f)

print(f"Encoding Complete. {len(encodeListKnown)} facial profiles stored.")