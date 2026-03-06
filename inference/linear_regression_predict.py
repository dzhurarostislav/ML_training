import numpy as np
import os
from architectures.linear_regression import LinearRegression

def run_inference(new_data):
    # 1. Загружаем единый архив состояния
    filepath = os.path.join('weights', 'linear_regression_model_state.npz')
    if not os.path.exists(filepath):
        raise FileNotFoundError("Свиток с весами не найден. Сначала обучи модель.")
        
    state = np.load(filepath)
    
    # 2. Инициализируем пустую фигуру на доске
    # Берем размерность входа прямо из формы сохраненных весов
    input_dim = state['W'].shape[0]
    model = LinearRegression(input_dim=input_dim)
    
    # 3. Внедряем память (веса)
    model.W = state['W']
    model.b = state['b']
    
    # 4. Извлекаем параметры реальности (скейлинг)
    X_mean = state['X_mean']
    X_std = state['X_std']
    y_mean = state['y_mean']
    y_std = state['y_std']
    
    print(f"Получены новые данные для анализа: {new_data.flatten()}")
    
    # --- ПРОЦЕСС ПРЕДСКАЗАНИЯ ---
    
    # Шаг А: Искажаем входные данные до масштабов матрицы (Скейлинг)
    # Защита от деления на ноль, если std == 0
    safe_X_std = np.where(X_std == 0, 1.0, X_std) 
    X_scaled = (new_data - X_mean) / safe_X_std
    
    # Шаг Б: Пропускаем через модель
    y_pred_scaled = model.forward(X_scaled)
    
    # Шаг В: Возвращаем ответ в реальный мир (Денормализация)
    y_pred_real = y_pred_scaled * y_std + y_mean
    
    print(f"Вердикт модели: {y_pred_real.flatten().round(2)}")
    return y_pred_real

if __name__ == "__main__":
    # Генерируем "новые" сырые данные из реального мира (например, 3 случайных числа)
    new_X = np.array([[5.0], [-2.5], [10.0]])
    run_inference(new_X)