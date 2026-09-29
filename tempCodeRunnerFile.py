import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import csv
from datetime import datetime
import os
from collections import defaultdict

# Load model
model = load_model("model/face_model.h5")
with open("model/labels.pkl", "rb") as f:
    le = pickle.load(f)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def classroom_mode():
    print("Starting Classroom Attendance... Press Q to Stop")

    marked_students = set()
    today = datetime.now().strftime("%Y-%m-%d")
    file = f"classroom_attendance_{today}.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Time"])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))
            face = face.reshape(1,100,100,1)/255.0

            prediction = model.predict(face)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.80:
                name = le.inverse_transform([class_id])[0]

                if name not in marked_students:
                    marked_students.add(name)
                    with open(file, "a", newline="") as f:
                        csv.writer(f).writerow([name, datetime.now().strftime("%H:%M:%S")])
                    print(f"{name} marked present")

                color = (0,255,0)
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Classroom Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def organization_mode():
    print("Starting Organization Entry Gate... Press Q to Stop")

    marked_today = set()
    detection_counter = defaultdict(int)

    today = datetime.now().strftime("%Y-%m-%d")
    file = f"organization_attendance_{today}.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            csv.writer(f).writerow(["Employee", "Entry Time"])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))
            face = face.reshape(1,100,100,1)/255.0

            prediction = model.predict(face)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.85:
                name = le.inverse_transform([class_id])[0]
                detection_counter[name] += 1

                if detection_counter[name] >= 5 and name not in marked_today:
                    marked_today.add(name)
                    with open(file, "a", newline="") as f:
                        csv.writer(f).writerow([name, datetime.now().strftime("%H:%M:%S")])
                    print(f"{name} ENTRY RECORDED")

                color = (0,255,0)
                text = name
            else:
                text = "ACCESS DENIED"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, text, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Organization Entry Gate", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# MAIN MENU
print("Select Mode:")
print("1 - Classroom Student Attendance")
print("2 - Organization Employee Entry")

choice = input("Enter choice (1/2): ")

if choice == "1":
    classroom_mode()
elif choice == "2":
    organization_mode()
else:
    print("Invalid Choice")