import gc  # Сборщик мусора для очистки памяти
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="РЖЯ Переводчик", page_icon="🖐️")

st.title("🖐️ Переводчик жестов (РЖЯ)")
st.write("Выберите способ загрузки изображения:")


# --- ОПТИМИЗАЦИЯ 1: Кэширование модели ---
# @st.cache_resource говорит Streamlit: "Загрузи это один раз и держи в памяти"
@st.cache_resource
def load_model():
    try:
        model_dict = pickle.load(open("./model.p", "rb"))
        return model_dict["model"]
    except FileNotFoundError:
        return None


model = load_model()

if model is None:
    st.error("Ошибка: Файл модели 'model.p' не найден.")
    st.stop()

# Инициализация MediaPipe (тоже можно вынести, но MP плохо кэшируется, оставим так)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Вкладки
tab1, tab2 = st.tabs(["📁 Загрузить фото", "📷 Сделать фото"])

image_source = None

with tab1:
    uploaded_file = st.file_uploader("Выберите файл...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file

with tab2:
    camera_file = st.camera_input("Сделайте снимок")
    if camera_file is not None:
        image_source = camera_file

if image_source is not None:
    # 1. Открываем фото
    image = Image.open(image_source)

    # Показываем пользователю
    if image_source == uploaded_file:
        st.image(image, caption="Загруженное фото", use_container_width=True)

    # 2. Конвертация в массив
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # --- ОПТИМИЗАЦИЯ 2: Уменьшение размера (Resize) ---
    # Если картинка больше 1000 пикселей по ширине, уменьшаем её
    # Это КРИТИЧЕСКИ экономит память
    max_width = 800
    if img_array.shape[1] > max_width:
        scale_ratio = max_width / img_array.shape[1]
        new_height = int(img_array.shape[0] * scale_ratio)
        img_array = cv2.resize(img_array, (max_width, new_height))

    # 3. Распознавание
    results = hands.process(img_array)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        hand_landmarks = results.multi_hand_landmarks[0]

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        st.markdown(
            f"""
        <div style="text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #155724;">Распознанный жест:</h3>
            <h1 style="color: #155724; font-size: 72px; margin: 0;">{predicted_character}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Рука не обнаружена.")

    # --- ОПТИМИЗАЦИЯ 3: Принудительная очистка ---
    del img_array
    del image
    del results
    gc.collect()
