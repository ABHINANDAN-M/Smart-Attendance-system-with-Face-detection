# import os
# import cv2
# import numpy as np
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from sklearn.preprocessing import LabelEncoder

# dataset_path = "dataset"

# faces = []
# labels = []

# for person in os.listdir(dataset_path):
#     person_path = os.path.join(dataset_path, person)
#     for image in os.listdir(person_path):
#         img = cv2.imread(os.path.join(person_path, image), 0)
#         faces.append(img)
#         labels.append(person)

# faces = np.array(faces)
# faces = faces.reshape(-1, 100, 100, 1) / 255.0

# le = LabelEncoder()
# labels = le.fit_transform(labels)
# labels = to_categorical(labels)

# # CNN Model
# model = Sequential()

# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)))
# model.add(MaxPooling2D(2,2))

# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(2,2))

# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(labels.shape[1], activation='softmax'))

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(faces, labels, epochs=10)

# model.save("model/face_model.h5")

# print("Model Trained Successfully!")






# import os
# import cv2
# import numpy as np
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# data = []
# labels = []
# label_dict = {}
# label_id = 0

# for role in ["students", "employees"]:
#     role_path = f"dataset/{role}"
#     if not os.path.exists(role_path):
#         continue

#     for person in os.listdir(role_path):
#         person_path = os.path.join(role_path, person)
#         label_dict[label_id] = person

#         for img in os.listdir(person_path):
#             img_path = os.path.join(person_path, img)
#             image = cv2.imread(img_path, 0)
#             image = cv2.resize(image, (100, 100))
#             data.append(image)
#             labels.append(label_id)

#         label_id += 1

# data = np.array(data).reshape(-1, 100, 100, 1) / 255.0
# labels = to_categorical(labels)

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(len(label_dict), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(data, labels, epochs=10)

# model.save("face_model.h5")
# np.save("labels.npy", label_dict)

# print("Model trained successfully!")



import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = []
labels = []
label_dict = {}
label_id = 0

for role in ["students", "employees"]:
    role_path = f"dataset/{role}"
    if not os.path.exists(role_path):
        continue

    for person in os.listdir(role_path):
        person_path = os.path.join(role_path, person)
        label_dict[label_id] = person

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            image = cv2.imread(img_path, 0)
            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(label_id)

        label_id += 1

data = np.array(data).reshape(-1, 100, 100, 1) / 255.0
labels = to_categorical(labels)

# 🔥 Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# 🔥 Stronger CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    datagen.flow(data, labels, subset='training', batch_size=16),
    validation_data=datagen.flow(data, labels, subset='validation'),
    epochs=25
)

model.save("face_model.keras")
np.save("labels.npy", label_dict)

print("✅ Model trained successfully!")
