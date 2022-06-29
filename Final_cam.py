# pip install numpy==1.20.1
# pip install opencv-python
# pip install tensorflow

import numpy as np
import cv2
import datetime
from threading import Thread
import tensorflow as tf
import os

# global variables
stop_thread = False             # controls thread execution
img = None                      # stores the image retrieved by the camera

# Loading the age model
direct = os.getcwd()
rel_path = "model_age"
abs_path = os.path.join(direct, rel_path)
age_model = tf.keras.models.load_model(abs_path)

# Loading the gender model
rel_path = "model_gender"
abs_path = os.path.join(direct, rel_path)
gender_model = tf.keras.models.load_model(abs_path)

# Loading the emotion model
rel_path = "model_emotion"
abs_path = os.path.join(direct, rel_path)
emotion_model = tf.keras.models.load_model(abs_path)

# Built in haar cascade for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# Global storage of face results
result = []
curr_result = []


# Thread for detecting and predicting age and gender of people on webcam screen
def predict_image_thread():

    global result

    while True:  # Loop until user presses escape

        age_dict = {
            0: "(0, 2)",
            1: "(4, 6)",
            2: "(8, 12)",
            3: "(15, 20)",
            4: "(25, 32)",
            5: "(38, 43)",
            6: "(48, 53)",
            7: "(60, 100)"
        }

        gender_dict = {
            0: "Female",
            1: "Male"
        }

        emotion_dict = {
            0: "Neutral", 
            1: "Happy", 
            2: "Sad", 
            3: "Surprised", 
            4: "Anger"
        }

        # Detect face
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        curr_result = []

        # Store predicted age, gender and position of each detected face
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]

            correct_face = cv2.resize(face, (224, 224))

            curr_face = np.expand_dims(correct_face, axis=0)

            age_pred = age_model.predict(curr_face)
            gender_pred = gender_model.predict(curr_face)
            emotion_pred = emotion_model.predict(curr_face)

            curr_result.append([age_dict[np.argmax(age_pred)], gender_dict[np.argmax(gender_pred)], emotion_dict[np.argmax(emotion_pred)], x, y, w, h])

        result = curr_result

        if stop_thread:
            break


# Thread for continuously reading webcam data
def start_capture_thread(cap):
    global img, stop_thread

    # continuously read frames from the camera
    while True:
        _, img = cap.read()

        if stop_thread:
            break


# Main function
def main():

    global img, stop_thread, result

    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
    t = Thread(target=start_capture_thread, args=(cap,), daemon=True) # a deamon thread is killed when the application exits
    s = Thread(target=predict_image_thread, args=(), daemon=True)

    t.start()
    s.start()

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0
    cur_fps = 0

    while True:  # Loop until user presses escape

        # blocks until the entire frame is read
        frames += 1

        # measure runtime: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()

        # compute fps but avoid division by zero
        if elapsed_time != 0:
            cur_fps = np.around(frames / elapsed_time, 1)

        # draw FPS text and display image
        if img is not None:
            
            # Faces detected
            if len(result) != 0:

                # Draw bounding box and results around the detected face
                for test_result in result: 
                    cv2.rectangle(img, (test_result[3], test_result[4]), (test_result[3] + test_result[5], test_result[4] + test_result[6]), (255, 0, 0), 1)
                    cv2.putText(img, test_result[0] + " " + test_result[1], (test_result[3], test_result[4]), cv2.FONT_HERSHEY_SIMPLEX, test_result[6] / 200.0 * 1.2, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, test_result[2], (test_result[3], test_result[4] + test_result[6]), cv2.FONT_HERSHEY_SIMPLEX, test_result[6] / 200.0 * 1.2, (255, 0, 0), 1, cv2.LINE_AA)               

            cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("webcam", img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            stop_thread = True
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()