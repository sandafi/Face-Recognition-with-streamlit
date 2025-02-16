''''
Kalo mau dirun disini biar web camnya keliatan yaa sandaaaaaa!
Real Time Face Recogition
	==> Setiap wajah yang disimpan di direktori dataset/ harus memiliki ID numerik unik berupa bilangan bulat, 
    seperti 1, 2, 3, dan seterusnya.                      
	==> Model yang dihitung dengan LBPH (wajah yang telah dilatih) harus disimpan di direktori trainer/.
'''

import cv2
import numpy as np
import os

# Inisialisasi recognizer dan load model
if not os.path.exists('trainer/trainer.yml'):
    print("Error: Trainer file 'trainer/trainer.yml' not found.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haarcascade
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(haarcascade_path):
    print(f"Error: Haarcascade file not found at {haarcascade_path}.")
    exit()

faceCascade = cv2.CascadeClassifier(haarcascade_path)

# Inisialisasi font
font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi ID
id = 0

# Daftar nama sesuai ID
names = ['None', 'Fiona', 'Sanda', 'Bella']

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

cam.set(3, 640)  # Lebar video
cam.set(4, 480)  # Tinggi video

# Ukuran minimum jendela untuk mendeteksi wajah
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Validasi ID dan confidence
        if id < len(names) and confidence < 100:
            name = names[id]
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        # Tampilkan nama dan confidence
        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    # Keluar dengan tombol 'ESC'
    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # Tombol ESC
        break

# Bersihkan resource
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

