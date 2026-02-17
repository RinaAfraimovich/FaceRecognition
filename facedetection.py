import cv2
import os


# Функция загрузки классификатора для лиц
def download_classifier():
    """
    Возвращает путь к классификатору OpenCV для распознавания лиц.
    Классификатор автоматически загружается из библиотеки OpenCV.
    """
    haarcascade_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    
    # Если файла нет, сообщаем об этом и возвращаем полный путь
    if not os.path.exists(haarcascade_file):
        print("Файл классификатора лиц отсутствует!")
        
    return haarcascade_file


# Основная программа
def main():
    # Получаем путь к классификатору лиц
    classifier_path = download_classifier()
    
    # Создаем объект CascadeClassifier для распознавания лиц
    face_cascade = cv2.CascadeClassifier(classifier_path)
    
    # Проверяем, загрузился ли классификатор
    if face_cascade.empty():
        print("Ошибка: не удалось загрузить классификатор лиц!")
        return
    
    # Подключаемся к камере (0 - индекс первой подключенной камеры)
    cap = cv2.VideoCapture(0)
    
    # Проверяем успешность подключения
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру!")
        return
    
    try:
        while True:
            # Захватываем кадр с камеры
            ret, frame = cap.read()
            
            # Обрабатываем ошибку захвата кадра
            if not ret:
                print("Ошибка: не удалось захватить кадр!")
                break
                
            # Конвертируем кадр в оттенки серого (лучше для распознавания)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Детектируем лица на изображении
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1, # Коэффициент масштабирования (уменьшение размера окна поиска)
                minNeighbors=5, # Минимальное число соседних областей для подтверждения нахождения лица
                minSize=(30, 30) # Минимальные размеры обнаруживаемого лица
            )
            
            # Отмечаем каждую найденную область лица зелёным прямоугольником
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Подписываем количество обнаруженных лиц
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Показываем обработанный кадр
            cv2.imshow('Face Detection', frame)
            
            # Ожидаем нажатия клавиши 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Завершаем работу с камерой и окнами
        cap.release()
        cv2.destroyAllWindows()
        print("Приложение завершило свою работу.")


if __name__ == "__main__":
    main()