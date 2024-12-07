# Import standar dan eksternal
import cv2
import numpy as np
import os
from controller import Robot, Display
from scanner import get_warped_document, resize_and_letter_box, segment_by_color
from itertools import count

# Konstanta
TIME_STEP = 100
HSV_LOW_RANGE = np.array([27, 0, 66])
HSV_UP_RANGE = np.array([180, 38, 255])
SAVE_TO_DISK = True  # Atur ke True untuk menyimpan gambar


# Fungsi counter untuk penamaan file output gambar
def counter(_count=count(1)):
    """Fungsi counter untuk penamaan file gambar"""
    return next(_count)


# Fungsi untuk menyimpan gambar ke disk
def save_image(image):
    """Menyimpan gambar ke folder 'Saved Pictures'"""
    save_path = r"C:\Users\mjibr\Pictures\Saved Pictures"  # Path folder tujuan
    if not os.path.exists(save_path):  # Cek jika folder tidak ada, buat folder
        os.makedirs(save_path)

    # Simpan gambar dengan penamaan unik
    file_name = f"{save_path}\\image_{counter()}.jpg"
    cv2.imwrite(file_name, image)


# Fungsi inisialisasi robot dan perangkat
def initialize():
    """Inisialisasi robot, kamera, dan display"""
    robot = Robot()

    camera = robot.getDevice("camera")
    camera.enable(100)  # Mengaktifkan kamera dengan waktu langkah 100ms

    display = robot.getDevice("image display")

    return robot, camera, display


# Fungsi untuk mengonversi gambar Webots menjadi numpy array
def webots_image_to_numpy(im, h, w):
    """Mengonversi gambar dari Webots ke numpy array"""
    return np.frombuffer(im, dtype=np.uint8).reshape((h, w, 4))


# Fungsi untuk menampilkan gambar numpy di display Webots
def display_numpy_image(im, display, display_width, display_height):
    """Menampilkan gambar numpy di display Webots"""
    display_image = display.imageNew(im.tobytes(), Display.BGRA, display_width, display_height)
    display.imagePaste(display_image, 0, 0, blend=False)
    display.imageDelete(display_image)


# Main program
if __name__ == "__main__":
    robot, camera, display = initialize()

    camera_width = camera.getWidth()
    camera_height = camera.getHeight()

    display_width = display.getWidth()
    display_height = display.getHeight()

    print(f"Camera HxW: {camera_height}x{camera_width}")
    print(f"Display HxW: {display_height}x{display_width}")

    while robot.step(TIME_STEP) != -1:
        # Mengambil gambar dari kamera
        webots_im = camera.getImage()
        numpy_im = webots_image_to_numpy(webots_im, camera_height, camera_width)

        # Jika SAVE_TO_DISK True, gambar akan disimpan
        if SAVE_TO_DISK:
            save_image(cv2.cvtColor(numpy_im, cv2.COLOR_RGB2BGR))  # Simpan gambar dalam format BGR

        # Melakukan segmentasi berdasarkan warna untuk mendeteksi dokumen
        mask = segment_by_color(numpy_im, HSV_LOW_RANGE, HSV_UP_RANGE)

        try:
            # Mencoba untuk mendeteksi dan mengoreksi perspektif dokumen
            document = get_warped_document(numpy_im, mask, debug=False)
            document = resize_and_letter_box(document, display_height, display_width)
        except ValueError as e:
            # Jika tidak ada dokumen yang terdeteksi, tampilkan gambar hitam
            document = np.zeros((display_width, display_height, 4), dtype=np.uint8)
            document[:, :, 0] = 255  # Menandakan kesalahan dalam deteksi dokumen
            print(e)

        # Menampilkan gambar yang telah diproses di display Webots
        display_numpy_image(document, display, display_width, display_height)

    robot.cleanup()
