import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from joblib import dump
import os

# Завантаження даних
try:
    ds = pd.read_csv("data/processed_train.csv")
    print("Дані успішно завантажені.")
except FileNotFoundError:
    print("Файл не знайдено. Перевірте шлях до файлу.")
    exit()

# Перевірка наявності необхідних колонок
if 'Price' not in ds.columns:
    print("Колонка 'Price' не знайдена у датасеті.")
    exit()

# Вибір ознак та цільової змінної
try:
    X = ds.drop(['Price', 'Unnamed: 0'], axis=1, errors='ignore')
    y = ds['Price']
except KeyError as e:
    print(f"Помилка у виборі колонок: {e}")
    exit()

# Поділ даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перевірка типу задачі
if y.nunique() > 10:  # Припущення: якщо багато унікальних значень, це регресія
    task_type = "regression"
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
else:
    task_type = "classification"
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

print(f"Визначено задачу: {task_type}")

# Тренування моделі
try:
    model.fit(X_train, y_train)
    print("Модель успішно натренована.")
except Exception as e:
    print(f"Помилка під час тренування моделі: {e}")
    exit()

# Оцінка моделі
if task_type == "classification":
    y_pred = model.predict(X_test)
    print("Класифікаційний звіт:\n")
    print(classification_report(y_test, y_pred))
else:  # regression
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2:.2f}")

# Збереження моделі
output_path = 'models/xgboost_model.pkl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
dump(model, output_path)
print(f"Модель збережено за адресою: {output_path}")
