''''
Menangkap beberapa Wajah dari banyak pengguna untuk disimpan di DataBase (direktori kumpulan data)
	==> Wajah akan disimpan di direktori: dataset/ (jika tidak ada, silakan buat satu)
	==> Setiap wajah akan memiliki ID bilangan bulat unik seperti 1, 2, 3, dst                     

'''

import cv2
import os

# Buka kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set ukuran video
cam.set(3, 640)  # lebar video
cam.set(4, 480)  # tinggi video

# Jalur file Haarcascade
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(haarcascade_path):
    print(f"Error: Haarcascade file not found at {haarcascade_path}.")
    exit()

# Inisialisasi face detector
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Input ID pengguna
face_id = input('\nEnter user ID and press <return>: ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")
count = 0

# Pastikan folder 'dataset' ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

while True:
    # Baca frame dari kamera
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Simpan gambar ke folder 'dataset'
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        # Tampilkan gambar
        cv2.imshow('image', img)

    # Keluar dengan tombol 'ESC'
    k = cv2.waitKey(100) & 0xFF
    if k == 27:  # Tombol 'ESC'
        break
    elif count >= 30:  # Ambil 30 sampel wajah, lalu berhenti
        break

# Bersihkan resource
print("\n[INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()



