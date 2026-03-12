import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("gender_model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera start
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if not ret :
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face,(100,100))
        face = face/255.0
        face = np.reshape(face,(1,100,100,3))

        prediction = model.predict(face)

        confidence = prediction[0][0] * 100

        if prediction[0][0] > 0.5:
            label = "Female"
            acc = confidence
        else:
            label = "Male"
            acc = 100 - confidence

        text = f"{label} {acc:.2f}%"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,text,(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Gender Detection",frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()