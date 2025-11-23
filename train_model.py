import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Загружаем данные
data_dict = pickle.load(open("./data.pickle", "rb"))

# Преобразуем в формат numpy (матрицы)
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Разделяем на обучение (80%) и тест (20%)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Создаем модель
model = RandomForestClassifier()

# Обучаем!
print("Обучаю модель...")
model.fit(x_train, y_train)

# Проверяем точность
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"{score * 100}% образцов классифицировано верно!")

# Сохраняем готовую модель
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()

print("Модель сохранена в model.p")
