from architectures.linear_regression import LinearRegression
from data.linear_regression_task import generate_linear_data
from tools import xavier_init
from tools import z_score_normalize
import numpy as np


def run_training():
    # 1. Гиперпараметры (настройка нашего "инструмента")
    LR = 1
    EPOCHS = 1000
    N_SAMPLES = 500

    # 2. Подготовка данных
    X, y, w_true, b_true = generate_linear_data(n_samples=N_SAMPLES)
    # Ау-ау, не забудь про скейлинг! 
    X_scaled = z_score_normalize(X)

    X_std = np.std(X, axis=0)

    # Скейлим игреки
    # Важно: нам нужно сохранить параметры скейлинга (mean и std) для инференса!
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std

    # 3. Инициализация модели
    model = LinearRegression(input_dim=X.shape[1])
    model.init_params(xavier_init)

    # 4. Основной цикл (The Arena)
    for epoch in range(EPOCHS):
        # --- Forward Pass ---
        y_pred = model.forward(X_scaled)
        
        # --- Считаем ошибку (Loss) ---
        # Напиши здесь формулу MSE через np.mean
        loss = np.mean((y_pred - y_scaled)**2)
        
        # --- Backward Pass ---
        # Считаем градиент ошибки (производная MSE по y_pred)
        # d_out = 2/m * (y_pred - y_true)
        m = X_scaled.shape[0]
        d_out = (2 / m) * (y_pred - y_scaled)
        
        dW, db = model.backward(d_out)
        
        # --- Update Weights ---
        # Здесь мы вручную обновляем параметры модели
        model.W -= LR * dW
        model.b -= LR * db

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

    # Когда "лев" закончил охоту и веса обучены:
    model.save_state(
        X_mean=np.mean(X, axis=0), 
        X_std=np.std(X, axis=0),
        y_mean=y_mean, 
        y_std=y_std
    )
    
    # Сравниваем с истинными весами из генератора
    print(f"Истинный W: {w_true.flatten()}")
    print(f"Обученный W: {model.W.flatten()}")

    print("\n--- Проверка ответов (Первые 5 примеров) ---")
    # Берем предсказания для первых пяти строк
    sample_pred_scaled = model.forward(X_scaled[:5])
    
    # Твоя магия денормализации в действии
    sample_pred_real = sample_pred_scaled * y_std + y_mean
    real_answers = y[:5]

    print(f"Предсказания модели: {sample_pred_real.flatten().round(2)}")
    print(f"Реальные ответы:     {real_answers.flatten().round(2)}")

    # Восстанавливаем реальный вес из матрицы
    W_reconstructed = model.W * (y_std / X_std)
    
    print(f"Истинный W: {w_true.flatten().round(4)}")
    print(f"Сырой обученный W (корреляция): {model.W.flatten().round(4)}")
    print(f"Восстановленный W (в реальном мире): {W_reconstructed.flatten().round(4)}")

if __name__ == "__main__":
    run_training()
