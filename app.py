import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Настройка страницы (заголовок и иконка)
st.set_page_config(page_title="РЖЯ Переводчик", page_icon="🖐️")

st.title("🖐️ Переводчик жестов (РЖЯ)")
st.write("Загрузите фото жеста, и нейросеть определит букву.")

# Загружаем модель
# try-except блок на случай, если модели нет
try:
    model_dict = pickle.load(open("./model.p", "rb"))
    model = model_dict["model"]
except FileNotFoundError:
    st.error(
        "Ошибка: Файл модели 'model.p' не найден. Сначала запустите train_model.py!"
    )
    st.stop()

# Настройка MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Словарь для перевода (если папки названы латиницей, а вывод нужен кириллицей)
# Если у тебя папки уже названы русскими буквами (А, Б...), этот словарь не обязателен,
# но он полезен, если нейросеть выдает "A" (английскую), а ты хочешь писать "А" (русскую).
# Можно оставить как есть, модель вернет то название папки, на которой училась.

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Отображаем фото
    image = Image.open(uploaded_file)
    # ВОТ ИСПРАВЛЕНИЕ: use_container_width=True вместо use_column_width=True
    st.image(image, caption="Загруженное фото", use_container_width=True)

    # 2. Подготовка картинки
    img_array = np.array(image)

    # Если есть альфа-канал (прозрачность), убираем его
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # 3. Распознавание
    results = hands.process(img_array)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        hand_landmarks = results.multi_hand_landmarks[0]

        # Сбор координат
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Нормализация
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # 4. Предсказание нейросети
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        # 5. Красивый вывод результата
        st.markdown(
            f"""
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
            <h3 style="color: #333;">Распознанный жест:</h3>
            <h1 style="color: #0068c9; font-size: 72px;">{predicted_character}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.warning(
            "⚠️ Рука не обнаружена. Попробуйте фото с другим освещением или фоном."
        )
