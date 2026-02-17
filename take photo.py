import cv2
import os
import keyboard

def create_folder_structure(base_folder, person_name):
    """
    Создает структуру папок: dataset/имя_человека/
    """
    # Создаем путь к папке человека
    person_folder = os.path.join(base_folder, person_name)
    
    # Создаем папку, если она не существует
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Создана папка: {person_folder}")
    else:
        print(f"Папка уже существует: {person_folder}")
    
    return person_folder

def get_next_image_number(folder):
    """
    Определяет следующий номер для изображения в папке
    """
    existing_files = os.listdir(folder)
    image_files = [f for f in existing_files if f.startswith('img') and f.endswith('.jpg')]
    
    if not image_files:
        return 1
    
    # Извлекаем номера из имен файлов
    numbers = []
    for file in image_files:
        try:
            # Извлекаем число из имени файла (img1.jpg -> 1)
            number = int(file[3:-4])
            numbers.append(number)
        except ValueError:
            continue
    
    if numbers:
        return max(numbers) + 1
    else:
        return 1

def main():
    print("=== Программа для фотосъемки ===")
    print("Нажмите ПРОБЕЛ для создания фотографии")
    print("Нажмите ESC для выхода из программы")
    print("-" * 40)
    
    # Запрашиваем имя человека
    person_name = input("Введите имя человека (для создания подпапки): ").strip()
    
    if not person_name:
        print("Имя не может быть пустым. Программа завершена.")
        return
    
    # Базовая папка для всех снимков
    base_folder = "C:\\DBFaces"
    
    # Создаем структуру папок
    person_folder = create_folder_structure(base_folder, person_name)
    
    # Инициализируем камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    print("\nКамера запущена. Ожидание команд...")
    
    while True:
        # Захватываем кадр с камеры
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры")
            break
        
        # Отображаем видео с камеры
        cv2.imshow('Camera - Press SPACE to capture, ESC to exit', frame)
        
        # Проверяем нажатие клавиш
        key = cv2.waitKey(1) & 0xFF
        
        # Выход по ESC
        if key == 27:  # ESC
            print("Программа завершена пользователем")
            break
        
        # Съемка по ПРОБЕЛУ
        elif key == 32:  # ПРОБЕЛ
            # Определяем следующий номер для изображения
            img_number = get_next_image_number(person_folder)
            img_name = f"img{img_number}.jpg"
            img_path = os.path.join(person_folder, img_name)
            
            # Сохраняем изображение
            cv2.imwrite(img_path, frame)
            print(f"Фото сохранено: {img_path}")
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Проверяем наличие необходимых библиотек
    try:
        import cv2
        import keyboard
    except ImportError as e:
        print("Ошибка: Необходимо установить библиотеки")
        print("Установите их командой:")
        print("pip install opencv-python keyboard")
        print(f"Детали ошибки: {e}")
        exit()
    
    main()