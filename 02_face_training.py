''''
Melatih Banyak Wajah yang disimpan di Basis Data:
	==> Setiap wajah harus memiliki ID bilangan bulat unik seperti 1, 2, 3, dst                       
	==> Model komputasi LBPH akan disimpan di direktori trainer/. (jika tidak ada, silakan buat satu)
	==> untuk menggunakan PIL, instal pustaka bantal dengan "pip install Pillow"
'''

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Inisialisasi recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Jalur file Haarcascade
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(haarcascade_path):
    print(f"Error: Haarcascade file not found at {haarcascade_path}.")
    exit()

# Inisialisasi face detector
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    if not imagePaths:
        print(f"Error: No images found in the dataset path: {path}")
        exit()

    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

# Cek folder 'trainer' dan buat jika tidak ada
if not os.path.exists('trainer'):
    os.makedirs('trainer')

print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Simpan model ke file trainer.yml
recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

# Cetak jumlah wajah yang telah dilatih
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

