# import cv2
# import os

# student_name = input("Enter Student Name: ")

# dataset_path = "dataset/" + student_name
# os.makedirs(dataset_path, exist_ok=True)

# cap = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
#                                        'haarcascade_frontalface_default.xml')

# count = 0

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         count += 1
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (100, 100))
#         cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

#     cv2.imshow("Face Capture", frame)

#     if count >= 100:
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("Dataset collection completed!")









# import cv2
# import os

# student_name = input("Enter Student Name: ")

# dataset_path = "dataset/" + student_name
# os.makedirs(dataset_path, exist_ok=True)

# cap = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

# count = 0

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to grab frame")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     print("Faces found:", len(faces))  # debug

#     for (x, y, w, h) in faces:
#         count += 1
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (100, 100))
#         cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

#     cv2.imshow("Face Capture", frame)

#     if count >= 100:
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("Dataset collection completed!")




# import cv2
# import os

# # ✅ Use OpenCV built-in haarcascade path (FIXED ERROR)
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

# # Ask Role
# role = input("Enter Role (student/employee): ").strip().lower()

# if role not in ["student", "employee"]:
#     print("Invalid role! Please enter student or employee.")
#     exit()

# # Ask Name
# name = input("Enter Name: ").strip()

# # Create folder path
# dataset_path = f"dataset/{role}s/{name}"
# os.makedirs(dataset_path, exist_ok=True)

# # Start Camera
# cap = cv2.VideoCapture(0)

# count = 0
# print("📸 Capturing images... Look at camera. Press 'q' to stop.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to access camera")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     for (x, y, w, h) in faces:
#         count += 1

#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (100, 100))

#         cv2.imwrite(f"{dataset_path}/{count}.jpg", face)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#         cv2.putText(frame, f"Images: {count}", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#     cv2.imshow("Capture Faces", frame)

#     # Stop after 50 images OR press q
#     if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
#         break

# cap.release()
# cv2.destroyAllWindows()

# print(f"✅ Images captured successfully for {name} ({role})")



import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

role = input("Enter Role (student/employee): ").strip().lower()

if role not in ["student", "employee"]:
    print("Invalid role! Please enter student or employee.")
    exit()

name = input("Enter Name: ").strip()

dataset_path = f"dataset/{role}s/{name}"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("📸 Capturing images... Move face slightly. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        count += 1
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Images: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 120:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ {count} images captured successfully for {name} ({role})")
