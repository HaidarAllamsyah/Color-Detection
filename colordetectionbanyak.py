# cv2 (OpenCV) → digunakan untuk mengakses kamera, memproses gambar/video, mengubah warna, mencari kontur, menggambar kotak, dll.
# numpy (np) → digunakan untuk membuat array numerik seperti batas bawah/atas warna HSV.
import cv2
import numpy as np


# Bagian ini berisi list warna yang ingin dideteksi
# setiap item terdiri dari 4 komponen
# Lower HSV → batas bawah warna
# Upper HSV → batas atas warna
# Nama Warna (untuk ditampilkan di layar)
# Warna BGR → warna kotak/teks saat menggambar di frame

# Menggunakan HSV (Hue Saturation Value) lebih stabil dibanding RGB karena tidak terpengaruh pencahayaan
colors = [
    # Merah
    ([0, 50, 50], [10, 255, 255], 'Merah', (0, 0, 255)),
    ([170, 50, 50], [180, 255, 255], 'Merah', (0, 0, 255)),
    # Jingga
    ([10, 50, 50], [20, 255, 255], 'Jingga', (0, 165, 255)),
    # Kuning
    ([20, 50, 50], [30, 255, 255], 'Kuning', (0, 255, 255)),
    # Hijau
    ([30, 50, 50], [80, 255, 255], 'Hijau', (0, 255, 0)),
    # Biru
    ([100, 50, 50], [130, 255, 255], 'Biru', (255, 0, 0)),
    # Nila
    ([130, 50, 50], [145, 255, 255], 'Nila', (255, 100, 0)),
    # Ungu
    ([145, 50, 50], [170, 255, 255], 'Ungu', (255, 0, 255)),
    # Putih
    ([0, 0, 200], [180, 30, 255], 'Putih', (255, 255, 255)),
    # Hitam
    ([0, 0, 0], [180, 255, 50], 'Hitam', (0, 0, 0)),
]

# Mengaktifkan webcam utama
# Setiap frame video akan dibaca dari objek CAP
cap = cv2.VideoCapture(0)

# loop utama untuk real time detection
# loop akan terus bejalan sampai user menghentikan program
# ret → status pengambilan frame (True/False).
# frame → gambar/video yang sedang ditampilkan kamera.
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # openCV menggunakan BGR, bukan RGB
    # warna BGR dirubah menjadi HSV untuk mempermudah deteksi warna
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # loop untuk deteksi warna dari list yang sudah disediakan di atas
    for (lower, upper, name, color_bgr) in colors:
        # Mask menghasilkan gambar hitam putih:
        # Mask inilah yang digunakan untuk menemukan objek berwarna
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        # Kontur adalah batas dari objek berwarna yang ditemukan
        # RETR_EXTERNAL → hanya mengambil kontur terluar (lebih efisien).
        # CHAIN_APPROX_SIMPLE → kompresi titik kontur agar lebih ringan.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # filter untuk area kontur, untuk mneghindari deteksi noise kecil
            # hanya objek dengan area > 500 pixel yang dianggap valid
            if cv2.contourArea(cnt) > 5000:
                # Memberikan koordinat untuk otak pembatas pada objek warna
                x, y, w, h = cv2.boundingRect(cnt)

                # Gambar kotak dengan warna sesuai
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

                # Tambahkan teks dengan warna sesuai
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 3)

    # Menampikan hasil video dengan kotak dan label warna
    cv2.imshow("Color Detection", frame)

    # program akan berhenti jika user menekan tombol q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# melepas webcam agar bisa dipakai di aplikasi yang lain
cap.release()
# menutup windows dari OpenCV
cv2.destroyAllWindows()