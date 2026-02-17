import cv2
import numpy as np
import os
import time
from datetime import datetime

class FaceRecognizer:
    def __init__(self):
        # Инициализация детектора лиц и распознавателя
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = []
        self.label_names = {}
        self.is_trained = False
        
    def train_from_folder(self, folder_path):
        """
        Обучение модели на основе фотографий в папке
        Структура папки:
        - folder_path/
            - person1/
                - photo1.jpg
                - photo2.jpg
            - person2/
                - photo1.jpg
                - photo2.jpg
        """
        faces = []
        labels = []
        label_id = 0
        
        print("Начало обучения модели...")
        
        # Проходим по всем папкам (каждая папка - один человек)
        for person_name in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            self.label_names[label_id] = person_name
            print(f"Обработка: {person_name} (ID: {label_id})")
            
            # Обрабатываем все изображения в папке
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                # Пропускаем не-изображения
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Загружаем и преобразуем изображение
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Детекция лиц
                faces_rect = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

              
                # Если найдено лицо, добавляем в обучающую выборку
                for (x, y, w, h) in faces_rect:
                    face_roi = gray[y:y+h, x:x+w]
                    # Приводим к стандартному размеру
                    face_resized = cv2.resize(face_roi, (100, 100))
                    faces.append(face_resized)
                    labels.append(label_id)
            
            label_id += 1
        
        if len(faces) == 0:
            print("Не найдено лиц для обучения!")
            return False
        
        # Обучаем модель
        print(f"Найдено {len(faces)} лиц для обучения.")
        self.face_recognizer.train(faces, np.array(labels))
        self.is_trained = True
        
        # Сохраняем модель
        self.face_recognizer.save('face_model.yml')
        np.save('label_names.npy', self.label_names)
        
        print("Модель успешно обучена и сохранена!")
        return True
    
    def load_model(self):
        """Загрузка сохраненной модели"""
        try:
            self.face_recognizer.read('face_model.yml')
            self.label_names = np.load('label_names.npy', allow_pickle=True).item()
            self.is_trained = True
            print("Модель успешно загружена!")
            return True
        except:
            print("Не удалось загрузить модель. Требуется обучение.")
            return False
    
    def recognize_from_camera(self):
        """Распознавание лиц с камеры"""
        if not self.is_trained:
            print("Модель не обучена! Сначала выполните обучение.")
            return
        
        # Открываем камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру!")
            return
        
        print("Запуск распознавания. Нажмите 'q' для выхода.")
        
        # Создаем папку для логирования
        log_dir = "recognition_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Создаем лог-файл
        log_file = open(os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), "w")
        
        last_recognition_time = 0
        recognition_cooldown = 2  # секунды между распознаваниями
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры")
                break
            
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Детекция лиц
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            current_time = time.time()
            
            for (x, y, w, h) in faces:
                # Выделяем область лица
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Распознаем лицо
                label_id, confidence = self.face_recognizer.predict(face_resized)
                
                # Определяем имя и цвет рамки
                if confidence < 90:  # Порог уверенности
                    name = self.label_names.get(label_id, "Unknown")
                    color = (0, 255, 0)  # Зеленый - распознан
                    
                    # Логируем распознавание (не чаще чем раз в cooldown секунд)
                    if current_time - last_recognition_time > recognition_cooldown:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"{timestamp} - Распознан: {name} (Уверенность: {confidence:.1f})\n"
                        print(log_entry.strip())
                        log_file.write(log_entry)
                        log_file.flush()
                        last_recognition_time = current_time
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Красный - не распознан
                
                # Рисуем прямоугольник вокруг лица
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Отображаем имя
                cv2.putText(frame, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Отображаем уверенность
                cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Отображаем FPS
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Показываем кадр
            cv2.imshow('Face Recognition', frame)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print("Распознавание завершено.")

def main():
    recognizer = FaceRecognizer()
    
    print("=" * 50)
    print("РАСПОЗНАВАНИЕ ЛИЦ С КАМЕРЫ")
    print("=" * 50)
    print("\nВыберите действие:")
    print("1. Обучить модель на основе папки с фотографиями")
    print("2. Загрузить сохраненную модель")
    print("3. Выход")
    
    choice = input("\nВведите номер действия (1-3): ")
    
    if choice == '1':
        folder_path = input("Введите путь к папке с фотографиями: ")
        if os.path.exists(folder_path):
            if recognizer.train_from_folder(folder_path):
                recognizer.recognize_from_camera()
        else:
            print("Указанная папка не существует!")
    
    elif choice == '2':
        if recognizer.load_model():
            recognizer.recognize_from_camera()
    
    elif choice == '3':
        print("Выход из программы.")
        return
    
    else:
        print("Неверный выбор!")

if __name__ == "__main__":
    main()