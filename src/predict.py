import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження нових даних
try:
    new_input_data = pd.read_csv('data/new_input.csv')
    print("Нові дані успішно завантажені.")
except FileNotFoundError:
    print("Файл не знайдено. Перевірте шлях до файлу.")
    exit()

# Завантаження моделі
try:
    model = joblib.load('models/xgboost_model.pkl')
    print("Модель успішно завантажена.")
except FileNotFoundError:
    print("Модель не знайдена. Перевірте шлях до моделі.")
    exit()

# Видалення непотрібних стовпців, які не використовувались під час тренування
X_new = new_input_data.drop(columns=['Price', 'Unnamed: 0'], errors='ignore')

# Перевірка на наявність пропущених значень
if X_new.isnull().any().any():
    print("У нових даних є пропущені значення!")
    # Можна додати код для обробки пропущених значень (наприклад, заповнення середнім значенням або іншими підходами)
    X_new = X_new.fillna(X_new.mean())  # Проста заміна пропущених значень на середнє

# Передбачення
predictions = model.predict(X_new)
print("Прогнози регресії зроблено.")

# Збереження результатів
output = new_input_data.copy()
output['predictions'] = predictions
output.to_csv('data/predictions.csv', index=False)
print("Прогнози збережено у файлі data/predictions.csv")

# Виведення результатів регресії
if 'Price' in new_input_data.columns:
    real_labels = new_input_data['Price']
    mse = mean_squared_error(real_labels, predictions)
    r2 = r2_score(real_labels, predictions)
    print(f"R^2 Score: {r2:.2f}")
else:
    print("У файлі відсутні реальні мітки для оцінки.")
