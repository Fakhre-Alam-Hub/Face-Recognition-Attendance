import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
from datetime import datetime
from mysql import connector


now = datetime.now()
date = now.strftime('%Y-%m-%d')

path = 'known_images'
images = []
known_face_names = []
myList = os.listdir(path)

for img_name in myList:
    cur_img = cv2.imread(f'{path}/{img_name}')
    images.append(cur_img)
    known_face_names.append(os.path.splitext(img_name)[0].upper())

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

known_face_encodings = findEncodings(images)

#Function for entry of new record
def markAttendance(name, date):
    conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="attendance"
    )
    cur = conn.cursor()

    # Get all the data from database table
    select_stmt = "SELECT* from attendance_table"
    cur.execute(select_stmt)
    result = cur.fetchall()

    nameList = []
    dateList = []
    for line in result:
        nameList.append(line[0])  # extract name and store in nameList
        dateList.append(line[2])   # extract date and store in dateList

    print(nameList)
    print(dateList)

    now = datetime.now()
    tString = now.strftime('%H:%M:%S')
    insert_stmt = (
        "INSERT IGNORE INTO attendance_table(Name,Time,Date)"
        "VALUES (%s, %s, %s)"
    )
    data = (name, tString, date)

    # Check if name is present in nameList
    # If name is not present in nameList then insert data into table
    if name not in nameList:       
        cur.execute(insert_stmt, data)
        conn.commit()
        conn.close()

    # If name is present then check for date 
    # If date is not present in dateList then insert data into table
    else:
        if date not in dateList:
            cur.execute(insert_stmt, data)
            conn.commit()
            conn.close()


cap = cv2.VideoCapture(0)
process_this_frame = True

while True:
    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        cur_face_locations = face_recognition.face_locations(rgb_small_frame)
        cur_face_encodings = face_recognition.face_encodings(rgb_small_frame, cur_face_locations)

        face_names = []
        for face_encoding in cur_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index]< 0.50:
                name = known_face_names[best_match_index]
                markAttendance(name,date)
            else: 
                name = 'Unknown'

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(cur_face_locations, face_names):
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
cap.release()
cv2.destroyAllWindows()