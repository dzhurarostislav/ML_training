import numpy as np
import os

class LinearRegression:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.W = None
        self.b = None
        # Сюда мы будем сохранять промежуточные значения для backward pass
        self.X = None
        self.z = None

    def init_params(self, initializer_func):
        # Здесь мы вызовем твой "инструмент" из tools
        self.W, self.b = initializer_func(self.input_dim)

    def forward(self, X):
        # X имеет форму (m, input_dim), где m - размер батча
        self.X = X
        # Магия линейной алгебры: y = XW + b
        # Напиши здесь реализацию через np.dot или оператор @
        self.z = np.dot(self.X, self.W) + self.b
        return self.z

    def backward(self, d_out):
        # d_out - это градиент функции потерь по выходу нашей модели (y_pred - y_true)
        m = self.X.shape[0]
        
        # Считаем градиенты для весов и смещения
        # Вспоминаем матан: производная по W зависит от X
        self.dW = (1 / m) * np.dot(self.X.T, d_out)
        self.db = (1 / m) * np.sum(d_out, axis=0, keepdims=True)
        
        return self.dW, self.db
    
    def save_state(self, folder_path='weights', **kwargs):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        filepath = os.path.join(folder_path, 'linear_regression_model_state.npz')
        # Упаковываем веса и любые переданные параметры скейлинга в один файл
        np.savez(filepath, W=self.W, b=self.b, **kwargs)
        print(f"--- Фрагмент памяти запечатан в {filepath} ---")