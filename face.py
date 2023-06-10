import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Define a function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Modify the load_images function to crop faces and save them
def load_images(directory):
    faces = []
    labels = []
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                image = cv2.imread(image_path)

                # Detect faces in the image
                faces_detected = detect_faces(image)

                for (x, y, w, h) in faces_detected:
                    # Crop the face region
                    face = image[y:y+h, x:x+w]

                    # Resize the face to a fixed size
                    resized_face = cv2.resize(face, (128, 128))

                    # Save the resized face
                    face_path = os.path.join(root, f"{label}_{file}")
                    cv2.imwrite(face_path, resized_face)

                    # Append the face path and label to the lists
                    image_paths.append(face_path)
                    labels.append(label)

    return labels, image_paths

# The remaining code for training and face recognition
def train_face_recognition(directory):
    labels, image_paths = load_images(directory)

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Load the resized faces
    faces = []
    for path in image_paths:
        face = cv2.imread(path)
        faces.append(face)

    # Flatten and normalize the image data
    flattened_faces = np.array([face.flatten() for face in faces])
    flattened_faces = flattened_faces / 255.0  # Normalize pixel values to the range [0, 1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(flattened_faces, labels, test_size=0.2, random_state=42)

    # Train the model
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save the confusion matrix as a plot image
    classes = label_encoder.classes_
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, title='Confusion Matrix',
           ylabel='True label', xlabel='Predicted label')

    plt.savefig('confusion_matrix.png')
    plt.close(fig)

    return model, label_encoder, image_paths

def perform_face_recognition(model, label_encoder, image_paths):
    while True:
        # Randomly select an image path
        selected_image_path = np.random.choice(image_paths)

        # Load the original image
        selected_image = cv2.imread(selected_image_path)

        # Preprocess the image
        resized_image = cv2.resize(selected_image, (128, 128))
        flattened_image = resized_image.flatten() / 255.0  # Normalize pixel values to the range [0, 1]

        # Perform face recognition
        prediction = model.predict([flattened_image])[0]
        label = label_encoder.inverse_transform([prediction])[0]

        # Show the image with the predicted label in a green bounding box
        faces_detected = detect_faces(selected_image)
        if len(faces_detected) > 0:
            (x, y, w, h) = faces_detected[0]
            cv2.rectangle(selected_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(selected_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Face Recognition', selected_image)
        else:
            print("No faces detected in the image.")

        # Randomly select another image
        other_image_path = np.random.choice(image_paths)
        while other_image_path == selected_image_path:
            other_image_path = np.random.choice(image_paths)

        # Load the other image
        other_image = cv2.imread(other_image_path)

        # Preprocess the other image
        resized_other_image = cv2.resize(other_image, (128, 128))
        flattened_other_image = resized_other_image.flatten() / 255.0  # Normalize pixel values to the range [0, 1]

        # Perform face recognition on the other image
        other_prediction = model.predict([flattened_other_image])[0]
        other_label = label_encoder.inverse_transform([other_prediction])[0]

        # Show the other image with its predicted label in a green bounding box
        faces_detected = detect_faces(other_image)
        if len(faces_detected) > 0:
            (other_x, other_y, other_w, other_h) = faces_detected[0]
            cv2.rectangle(other_image, (other_x, other_y), (other_x+other_w, other_y+other_h), (0, 255, 0), 2)
            cv2.putText(other_image, other_label, (other_x, other_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Other Image', other_image)
        else:
            print("No faces detected in the other image.")

        # Wait for a button press
        key = cv2.waitKey(0)

        # Continue comparing faces if 'y' is pressed
        if key == ord('y'):
            continue
        # Stop the program if 'n' is pressed
        elif key == ord('n'):
            break

    cv2.destroyAllWindows()

# Usage example
data_directory = 'data'
model, label_encoder, image_paths = train_face_recognition(data_directory)
perform_face_recognition(model, label_encoder, image_paths)