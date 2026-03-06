import numpy as np

def generate_linear_data(n_samples=100, input_dim=1, w_true=None, b_true=None, noise_std=0.1):
    """
    Генерирует синтетические данные: y = Xw + b + noise
    """
    # Если истинные веса не заданы, создаем их случайно
    if w_true is None:
        w_true = np.random.randn(input_dim, 1) * 10
    if b_true is None:
        b_true = np.random.randn(1) * 5
        
    # Генерируем X (например, в диапазоне от -10 до 10)
    X = np.random.uniform(-10, 10, size=(n_samples, input_dim))
    
    # Считаем чистый результат
    y_clean = np.dot(X, w_true) + b_true
    
    # Добавляем шум (нормальное распределение с центром в 0)
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    
    y = y_clean + noise
    
    return X, y, w_true, b_true

if __name__ == "__main__":
    # Тестовый запуск для проверки в Antigravity
    X, y, w, b = generate_linear_data(n_samples=5, noise_std=0.5)
    print(f"Сгенерировали {X.shape[0]} примеров.")
    print(f"Истинные веса (то, что модель должна угадать): {w.flatten()}, b: {b}")