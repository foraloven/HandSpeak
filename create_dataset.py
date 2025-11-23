import os
import pickle

import cv2
import mediapipe as mp
import numpy as np

# --- НАСТРОЙКИ ---
DATA_DIR = "./data"

# Настройки MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# static_image_mode=True важно для фото
# min_detection_confidence=0.3 — порог уверенности
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []


# --- ГЛАВНАЯ ФИШКА: Функция для чтения путей с русскими буквами ---
def imread_safe(path):
    # Открываем файл средствами Python (он понимает Unicode/UTF-8)
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    # Декодируем массив байт в картинку
    return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)


# Проверка папки
if not os.path.exists(DATA_DIR):
    print(f"Папка {DATA_DIR} не найдена!")
    exit()

directories = os.listdir(DATA_DIR)
print(f"Найдено папок: {len(directories)}")

success_count = 0
fail_count = 0

for dir_ in directories:
    # Пропускаем системные файлы
    if dir_.startswith("."):
        continue

    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Обрабатываю папку: {dir_}")

    files = os.listdir(dir_path)

    for img_path in files:
        # Пропускаем, если это не картинка
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.join(DATA_DIR, dir_, img_path)

        # ИСПОЛЬЗУЕМ БЕЗОПАСНОЕ ЧТЕНИЕ
        img = imread_safe(full_path)

        if img is None:
            print(f"  ❌ Не удалось прочитать: {img_path}")
            fail_count += 1
            continue

        # Конвертация в RGB (MediaPipe требует RGB)
        # Иногда imdecode возвращает 4 канала (RGBA), если есть прозрачность
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Попытка найти руку
        results = hands.process(img_rgb)

        # Если не нашли, пробуем повернуть (частая проблема телефонов)
        if not results.multi_hand_landmarks:
            img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            # Берем первую найденную руку
            hand_landmarks = results.multi_hand_landmarks[0]

            # Собираем координаты
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Нормализация (сдвигаем к нулю)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            success_count += 1
        else:
            fail_count += 1

# Сохраняем
if success_count > 0:
    f = open("data.pickle", "wb")
    pickle.dump({"data": data, "labels": labels}, f)
    f.close()
    print("\n================ ГОТОВО ================")
    print(f"Успешно обработано: {success_count} фото")
    print(f"Не распознано рук: {fail_count} фото")
    print("Теперь запускай train_model.py!")
else:
    print("\n❌ Ошибка: Не удалось извлечь данные ни из одного фото.")
