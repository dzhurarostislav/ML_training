import numpy as np

def xavier_init(shape):
    """
    Инициализация Ксавье (Glorot). 
    Подходит для симметричных активаций (Tanh, Sigmoid).
    """
    if isinstance(shape, int):
        n_in = shape
        n_out = 1
        actual_shape = (n_in, n_out)
    else:
        n_in = shape[0]
        n_out = shape[1] if len(shape) > 1 else 1
        actual_shape = shape
    # Дисперсия = 2 / (n_in + n_out)
    limit = np.sqrt(6 / (n_in + n_out))
    weights = np.random.uniform(-limit, limit, size=actual_shape)
    bias = np.zeros((1, n_out))
    return weights, bias

def he_init(shape):
    """
    Инициализация Хэ (Kaiming).
    Идеальна для ReLU.
    """
    n_in = shape[0]
    n_out = shape[1] if len(shape) > 1 else 1
    # Стандартное отклонение = sqrt(2 / n_in)
    std = np.sqrt(2.0 / n_in)
    weights = np.random.randn(*shape) * std
    bias = np.zeros((1, n_out))
    return weights, bias

def min_max_scale(data, min_val=-1, max_val=1):
    """Приводит матрицу к диапазону [min_val, max_val]"""
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # Чтобы не делиться на ноль, если данные одинаковые
    denom = data_max - data_min
    denom[denom == 0] = 1.0
    
    std_data = (data - data_min) / denom
    scaled_data = std_data * (max_val - min_val) + min_val
    return scaled_data

def z_score_normalize(data):
    """
    Стандартизация: среднее = 0, std = 1.
    Формула: (x - mean) / std
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Защита от деления на ноль (если все значения в колонке одинаковые)
    std[std == 0] = 1.0
    
    return (data - mean) / std