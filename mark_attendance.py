import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
import os
import time

# --- 1. Load the AI Data ---
print("Loading Encoded Data...")
try:
    with open("EncodeFile.p", "rb") as f:
        encodeListKnown, classNames = pickle.load(f)
    print(f"Database Loaded: {len(classNames)} profiles found.")
except:
    print("Error: EncodeFile.p not found! Run face_encode.py first.")
    exit()

status_msg = ""
status_color = (0, 0, 0)
msg_expiry = 0

def markAttendance(name):
    global status_msg, status_color, msg_expiry
    now = datetime.now()
    date_str = now.strftime('%d-%m-%Y')
    time_str = now.strftime('%H:%M:%S')
    file_name = 'Attendance.csv'
    
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write("NAME,DATE,TIME\n")

    with open(file_name, 'r+') as f:
        lines = f.readlines()
        already_marked = any(name in line and date_str in line for line in lines)
        if not already_marked:
            f.writelines(f"{name},{date_str},{time_str}\n")
            status_msg = f"SUCCESS: {name} MARKED"
            status_color = (0, 255, 0)
            msg_expiry = time.time() + 3.0 
        else:
            status_msg = f"{name} ALREADY MARKED TODAY"
            status_color = (0, 165, 255)
            msg_expiry = time.time() + 1.5

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame, model="large")

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Sort distances to find the two closest matches
        sorted_indices = np.argsort(faceDis)
        best_match_idx = sorted_indices[0]
        
        best_dist = faceDis[best_match_idx]
        
        # LOGIC UPDATE:
        # 1. Best distance must be very strict (0.42)
        # 2. If there's a second person in the database, the best match must be 
        #    at least 0.05 units "better" than the second best.
        is_certain = False
        if best_dist < 0.42:
            if len(faceDis) > 1:
                second_best_dist = faceDis[sorted_indices[1]]
                if (second_best_dist - best_dist) > 0.05:
                    is_certain = True
            else:
                is_certain = True

        if is_certain:
            name = classNames[best_match_idx].upper()
            markAttendance(name)
            box_color = (0, 255, 0) 
        else:
            name = "UNKNOWN / RE-VERIFY"
            box_color = (0, 0, 255) 

        y1, x2, y2, x1 = [v * 4 for v in faceLoc]
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, box_color, 2)

    if time.time() < msg_expiry:
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), status_color, cv2.FILLED)
        cv2.putText(img, status_msg, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('AI Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()