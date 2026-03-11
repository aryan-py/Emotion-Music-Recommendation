import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv",
}

# Tracks the most recently detected emotion index; default Neutral
current_emotion_idx = [4]


def music_rec():
    df = pd.read_csv(music_dist[current_emotion_idx[0]])
    df = df[['Name', 'Album', 'Artist']]
    return df.head(15)


def process_frame(image_bytes):
    """Decode a JPEG frame sent from the browser, run face detection and
    emotion classification, return (annotated_jpeg_bytes, emotion_str, songs_df).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, emotion_dict[current_emotion_idx[0]], music_rec()

    img = cv2.resize(img, (600, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

    for (x, y, w, h) in face_rects:
        cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped)
        maxindex = int(np.argmax(prediction))
        current_emotion_idx[0] = maxindex
        cv2.putText(img, emotion_dict[maxindex], (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes(), emotion_dict[current_emotion_idx[0]], music_rec()
