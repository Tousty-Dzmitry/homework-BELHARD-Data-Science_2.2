import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Загрузка датасета
def load_data(file):
    X = file.data
    y = file.target
    return X, y

# Разделение данных на обучающую и тестовую выборки, стратификация по y
def separation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

"""
 добавление параметра stratify=y, который гарантирует, что распределение классов в y_train и y_test 
 будет пропорционально распределению классов в исходном массиве y. Это особенно полезно для 
 несбалансированных датасетов, чтобы сохранить соотношение классов в обеих выборках.
"""
# Создание и обучение модели логистической регрессии
def education(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Оценка точности модели
def model_accuracy(yt, yp):
    return accuracy_score(yt, yp)