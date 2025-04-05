from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# матрица ошибок


def plot_confusion_matrix(y_true, y_pred, class_labels=None):
    """
    Визуализирует матрицу ошибок
    
    Параметры:
        y_true: истинные метки
        y_pred: предсказанные метки
        class_labels: список названий классов (опционально)
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()



# Распределение классов

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import pandas as pd

def distribution(data):
    # Загрузка данных
    digits = data

    # Создаем DataFrame с метками цифр
    df = pd.DataFrame({'digit': digits.target})

    # Построение распределения цифр
    plt.figure(figsize=(10, 6))

# Вариант 1: Без hue (простой вариант)
# sns.countplot(data=df, x='digit', color='royalblue')

    # Вариант 2: С hue (рекомендуемый способ)
    sns.countplot(data=df, x='digit', hue='digit', palette='viridis', legend=False)

    plt.title('Распределение цифр в датасете', pad=20)
    plt.xlabel('Цифра')
    plt.ylabel('Количество образцов')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# 3D матрица

# Убедитесь, что это ПЕРВАЯ строка в ячейке
# %matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
    


def volume_matrix(data):
    
    # Загрузка данных
    digits = data
    X = digits.data
    y = digits.target
    
    # Выполнение t-SNE преобразования
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)  # Вот где создается X_tsne!
    
    # Построение 3D-графика
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Визуализация (теперь X_tsne существует)
    scatter = ax.scatter(
        X_tsne[:, 0], 
        X_tsne[:, 1], 
        X_tsne[:, 2],
        c=y, 
        cmap='tab10',
        s=20,
        alpha=0.8
    )
    
    # Настройки графика
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title('3D проекция цифр с t-SNE')
    
    # Цветовая легенда
    plt.colorbar(scatter, label='Цифры')
    plt.show()