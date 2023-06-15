import os
import random
import shutil
import face_recognition
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

root_dir = os.getcwd()
# Set the path to the main directory
data_dir = f'{root_dir}/data'
no_face_dir = os.path.join(root_dir, 'no-face')
processed_dir = os.path.join(root_dir, 'processed')
if not os.path.exists(no_face_dir):
    os.makedirs(no_face_dir)
    print(f"Directory '{no_face_dir}' created.")
else:
    print(f"Directory '{no_face_dir}' already exists.")

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    print(f"Directory '{processed_dir}' created.")
else:
    print(f"Directory '{processed_dir}' already exists.")

# Set the target size for resizing
target_size = (224, 224)

# Set the train, val, and test ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Initialize lists to store labels and face encodings
known_labels = []
known_encodings = []

# Loop through each subdirectory
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)

    # Resize and replace the images in the subdirectory
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)

        # Resize the image and overwrite it with the resized version
        image = Image.open(image_path)
        resized_image = image.resize(target_size)
        resized_image.save(image_path)
        print(f"Image {image_path} resized and saved")

        # Load the resized image and encode the face
        resized_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(resized_image)

        if len(face_locations) == 0:
            # No face detected, move the image to the "no-face" directory
            shutil.move(image_path, os.path.join(no_face_dir, image_file))
            print(f"No face detected in {image_file}")
        else:
            encoding = face_recognition.face_encodings(resized_image)[0]

            # Append the label and encoding to the respective lists
            known_labels.append(label)
            known_encodings.append(encoding)

# Split the dataset into train, val, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    known_encodings, known_labels, test_size=test_ratio, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

# Create train, val, and test directories
train_dir = os.path.join(processed_dir, 'train')
val_dir = os.path.join(processed_dir, 'val')
test_dir = os.path.join(processed_dir, 'test')
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Move images to train, val, and test directories
for i, encoding in enumerate(X_train):
    label = y_train[i]
    label_dir = os.path.join(train_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    label_files = os.listdir(os.path.join(data_dir, label))
    if i < len(label_files):
        image_file = label_files[i]
        shutil.copy(os.path.join(data_dir, label, image_file), os.path.join(label_dir, image_file))
    else:
        print(f"Image file not found for label {label} at index {i} in train data.")

for i, encoding in enumerate(X_val):
    label = y_val[i]
    label_dir = os.path.join(val_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    label_files = os.listdir(os.path.join(data_dir, label))
    if i < len(label_files):
        image_file = label_files[i]
        shutil.copy(os.path.join(data_dir, label, image_file), os.path.join(label_dir, image_file))
    else:
        print(f"Image file not found for label {label} at index {i} in val data.")

for i, encoding in enumerate(X_test):
    label = y_test[i]
    label_dir = os.path.join(test_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    label_files = os.listdir(os.path.join(data_dir, label))
    if i < len(label_files):
        image_file = label_files[i]
        shutil.copy(os.path.join(data_dir, label, image_file), os.path.join(label_dir, image_file))
    else:
        print(f"Image file not found for label {label} at index {i} in test data.")

# Implement image augmentation on the train data
train_aug_dir = os.path.join(processed_dir, 'train_aug')
if not os.path.exists(train_aug_dir):
    os.makedirs(train_aug_dir)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True)

for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        img = Image.open(image_path)
        img_array = np.array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        save_dir = os.path.join(train_aug_dir, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 50:  # Generate 50 augmented images per original image
                break

# Load MobileNet model without top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for classification
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Update the number of units to match the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_aug_dir, target_size=target_size,
                                                    batch_size=32, class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=target_size,
                                                batch_size=32, class_mode='categorical')

# Train the model
history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator),
                              epochs=32, validation_data=val_generator, validation_steps=len(val_generator))

# Save the model
model.save('face-mobile.h5')

# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=target_size,
                                                  batch_size=32, class_mode='categorical', shuffle=False)

y_true = test_generator.classes
y_pred = model.predict_generator(test_generator).argmax(axis=-1)
accuracy = accuracy_score(y_true, y_pred)
confusion_mat = confusion_matrix(y_true, y_pred)

# Plot the accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')
plt.close()

# Plot the confusion matrix
labels = sorted(test_generator.class_indices.keys())
sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')
plt.close()