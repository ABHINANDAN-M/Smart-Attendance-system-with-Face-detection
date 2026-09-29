# import cv2
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import os
# from datetime import datetime

# marked_students = set()
# model = load_model("model/face_model.h5")

# dataset_path = "dataset"
# students = os.listdir(dataset_path)

# le = LabelEncoder()
# le.fit(students)

# cap = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
#                                       'haarcascade_frontalface_default.xml')

# attendance_list = []

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (100,100))
#         face = face.reshape(1,100,100,1)/255.0

#         prediction = model.predict(face)
#         label = le.inverse_transform([np.argmax(prediction)])[0]

#         cv2.putText(frame, label, (x,y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9, (0,255,0), 2)

#         if label not in attendance_list:
#             attendance_list.append(label)
#             time = datetime.now().strftime("%H:%M:%S")
#             with open("attendance.csv", "a") as f:
#                 f.write(f"{label},{time}\n")

#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

#     cv2.imshow("Attendance System", frame)

#     if cv2.waitKey(1) == 13:
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import csv
# import os
# from datetime import datetime
# from tensorflow.keras.models import load_model

# # ✅ FIXED Haarcascade Path
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

# # Load trained model
# model = load_model("face_model.h5")

# # Load label dictionary
# label_dict = np.load("labels.npy", allow_pickle=True).item()

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# marked_names = set()

# def detect_role(name):
#     if os.path.exists(f"dataset/students/{name}"):
#         return "student"
#     elif os.path.exists(f"dataset/employees/{name}"):
#         return "employee"
#     return "unknown"

# def mark_attendance(name):
#     role = detect_role(name)
#     filename = f"{role}_attendance.csv"

#     if name not in marked_names:
#         with open(filename, "a", newline="") as f:
#             writer = csv.writer(f)

#             # Add header if file empty
#             if os.stat(filename).st_size == 0:
#                 writer.writerow(["Name", "Time"])

#             writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

#         marked_names.add(name)
#         print(f"Marked attendance for {name}")

# print("Attendance system started... Press 'q' to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5,
#         minSize=(30,30)
#     )

#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (100,100))
#         face = face.reshape(1,100,100,1) / 255.0

#         prediction = model.predict(face, verbose=0)
#         confidence = np.max(prediction)
#         label = label_dict[np.argmax(prediction)]

#         if confidence > 0.75:
#             text = label
#             color = (0,255,0)
#             mark_attendance(label)
#         else:
#             text = "Unknown"
#             color = (0,0,255)

#         cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
#         cv2.putText(frame,text,(x,y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

#     cv2.imshow("Attendance System",frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import csv
import os
from datetime import datetime
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model = load_model("face_model.keras")
label_dict = np.load("labels.npy", allow_pickle=True).item()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

marked_names = set()
prediction_buffer = {}

def detect_role(name):
    if os.path.exists(f"dataset/students/{name}"):
        return "student"
    elif os.path.exists(f"dataset/employees/{name}"):
        return "employee"
    return "unknown"

def mark_attendance(name):
    role = detect_role(name)
    filename = f"{role}_attendance.csv"

    if name not in marked_names:
        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["Name", "Time"])

            writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

        marked_names.add(name)
        print(f"✅ Attendance marked for {name}")

print("🚀 Attendance system started... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100,100))
        face = face.reshape(1,100,100,1) / 255.0

        prediction = model.predict(face, verbose=0)
        confidence = np.max(prediction)
        label = label_dict[np.argmax(prediction)]

        # 🔥 Anti-Fluctuation Buffer (stability system)
        if label not in prediction_buffer:
            prediction_buffer[label] = 0

        if confidence > 0.80:
            prediction_buffer[label] += 1
        else:
            prediction_buffer[label] = 0

        if prediction_buffer[label] >= 5:
            text = label
            color = (0,255,0)
            mark_attendance(label)
        else:
            text = "Scanning..."
            color = (0,165,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,f"{text} ({confidence:.2f})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Attendance System",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
