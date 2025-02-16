import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Load model LBPH jika tersedia
MODEL_PATH = 'trainer/trainer.yml'
if not os.path.exists(MODEL_PATH):
    st.error("Error: Trainer file 'trainer/trainer.yml' not found.")
    st.stop()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load Haarcascade
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(haarcascade_path)

# Daftar nama ID
names = ['None', 'Fiona', 'Sanda', 'Bella']

st.title("Real-Time Face Recognition with Streamlit")
st.write("Upload an image or capture from the webcam.")

# Input webcam
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert file to OpenCV image
    img = Image.open(img_file_buffer)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if id < len(names) and confidence < 100:
            name = names[id]
            confidence_text = f"Confidence: {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"Confidence: {round(100 - confidence)}%"

        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # Convert back to PIL image and show
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Detected Faces", use_column_width=True)
