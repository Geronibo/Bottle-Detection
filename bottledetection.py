import cv2
from ultralytics import YOLO
from collections import Counter, deque
import time
import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog,QLabel,QFrame
import sys
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

# Global değişkenler
object_counter = Counter()
object_positions = {}
object_visibility = {}
history_length = 40
max_invisible_frames = 30
last_print_time = 0
MODELL = "bestyy.pt"
model = YOLO(MODELL)  # Modeli baştan yükle


def find_available_camera():
    """Bağlı olan ve çalışan ilk kamerayı bulur."""
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera {i} bulundu, bağlanılıyor...")
            cap.release()
            return i
    print("Uygun kamera bulunamadı!")
    return None

def is_new_detection(class_name, x, y):
    """Bir nesnenin yeni bir tespit olup olmadığını kontrol eder."""
    global object_positions

    if class_name not in object_positions:
        object_positions[class_name] = deque(maxlen=history_length)

    for px, py in object_positions[class_name]:
        distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)
        if distance < 400:
            return False

    object_positions[class_name].append((x, y))
    return True

def save_results_to_file():
    """Son nesne sayımını bir dosyaya saat bilgisiyle kaydeder."""
    file_path = os.path.join(os.getcwd(), "tespitler.txt")  # Mevcut dizinde oluştur
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "a", encoding="utf-8") as file:  # Append modunda aç
        file.write(f"\n--- {now} ---\n")
        for obj, count in object_counter.items():
            file.write(f"{count} adet {obj}\n")

    print(f"Son tespitler '{file_path}' dosyasına kaydedildi.")

def bottle_detection():
    global last_print_time

    camera_index = find_available_camera()
    if camera_index is None:
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Kamera {camera_index} açılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                break

            results = model.predict(source=frame, conf=0.70, iou=0.45, max_det=10, stream=True, verbose=False)
            detected_objects = set()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence >= 0.70:
                        class_name = model.names[class_id]
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                        if is_new_detection(class_name, center_x, center_y):
                            if class_name not in object_visibility or not object_visibility.get(class_name, False):
                                object_counter[class_name] += 1
                                object_visibility[class_name] = True

                        detected_objects.add(class_name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} ({confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for class_name in list(object_visibility.keys()):
                if class_name not in detected_objects:
                    if object_visibility[class_name]:
                        object_visibility[class_name] = False
                elif not object_visibility[class_name]:
                    object_visibility[class_name] = True
                    object_counter[class_name] += 1

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_print_time >= 1:
                last_print_time = current_time
                print("")
                for obj, count in object_counter.items():
                    print(f"{count} adet {obj}")

            y_offset = 60
            for obj, count in object_counter.items():
                text = f"{count} adet {obj}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                
            frame = cv2.resize(frame, (1920, 1080))
            cv2.imshow("Nesne Tespiti", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('k'):
                save_results_to_file()
            elif key == ord('q'):
                print("Çıkış yapılıyor...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
def image_detection():
    """Seçilen bir resimde nesne tespiti yapar."""
    file_path, _ = QFileDialog.getOpenFileName(None, "Resim Seç", "", "Image Files (*.png *.jpg *.jpeg)")
    if not file_path:
        return  

    frame = cv2.imread(file_path)
    results = model.predict(frame, conf=0.70, iou=0.45, max_det=10)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence >= 0.90:
                class_name = model.names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Resim - Nesne Tespiti", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_detection():
    """Seçilen bir videoda nesne tespiti yapar."""
    file_path, _ = QFileDialog.getOpenFileName(None, "Video Seç", "", "Video Files (*.mp4 *.avi *.mov)")
    if not file_path:
        return  

    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.70, iou=0.45, max_det=10)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence >= 0.70:
                    class_name = model.names[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video - Nesne Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çık
            break

    cap.release()
    cv2.destroyAllWindows()

class ObjectDetectionApp(QWidget):
    """Ana arayüz uygulaması."""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Nesne Tespit Arayüzü")
        self.setGeometry(100, 100, 600, 500)
        self.setStyleSheet("background-color: #f4f4f4;")  # Açık gri arkaplan

        layout = QVBoxLayout()

        # Başlık
        title = QLabel("Nesne Tespit Uygulaması")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Çerçeve
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 15px;
                border: 2px solid #ccc;
                padding: 20px;
            }
        """)
        frame_layout = QVBoxLayout()

        # Buton Stili
        button_style = """
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """

        # Butonlar
        btn_webcam = QPushButton("📷 Webcam ile Çalıştır")
        btn_webcam.setStyleSheet(button_style)
        btn_webcam.clicked.connect(bottle_detection)
        frame_layout.addWidget(btn_webcam)

        btn_image = QPushButton("🖼️ Resim ile Çalıştır")
        btn_image.setStyleSheet(button_style)
        btn_image.clicked.connect(image_detection)
        frame_layout.addWidget(btn_image)

        btn_video = QPushButton("🎥 Video ile Çalıştır")
        btn_video.setStyleSheet(button_style)
        btn_video.clicked.connect(video_detection)
        frame_layout.addWidget(btn_video)

        frame.setLayout(frame_layout)
        layout.addWidget(frame)

        self.setLayout(layout)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())  