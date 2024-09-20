import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Заголовок приложения
st.title("Разработка логистической регрессии")

#Загрузка файла CSV
uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

if uploaded_file is not None:
    # Чтение данных из файла
    data = pd.read_csv(uploaded_file)
    st.write("Данные загружены:")
    st.dataframe(data)

    # Выбор целевой переменной
    target_column = st.selectbox("Выберите целевую переменную:", data.columns)

    # Выбор признаков для регрессии
    feature_columns = st.multiselect("Выберите признаки для регрессии:", data.columns.drop(target_column))

    if len(feature_columns) < 2:
        st.warning("Выберите как минимум два признака для построения графика.")
    else:
        # Нормализация значений признаков
        scaler = StandardScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])

        # Создание класса логистической регрессии
        class LogReg:
            def __init__(self, learning_rate=0.1, n_inputs=2):
                self.learning_rate = learning_rate
                self.n_inputs = n_inputs
                self.coef_ = np.zeros(n_inputs)
                self.intercept_ = 0
                self.n_epochs = 400

            def sigmoid(self, z):
                return 1 / (1 + np.exp(-z))

            def grad(self, X, y, w_coef, w_intercept):
                y_pred = self.sigmoid(X @ w_coef + w_intercept)
                error = y_pred - y
                dw = X.T @ error / len(y)
                db = np.sum(error) / len(y)
                return dw, db
            
            def fit(self, X, y):
                for i in range(1, self.n_epochs + 1):
                    dw, db = self.grad(X, y, self.coef_, self.intercept_)
                    self.coef_ -= self.learning_rate * dw
                    self.intercept_ -= self.learning_rate * db

            def predict(self, X):
                y_pred_prob = self.sigmoid(X @ self.coef_ + self.intercept_)
                return (y_pred_prob >= 0.5).astype(int)

        if st.button("Выполнить регрессию"):
            # Разделение данных на признаки и целевую переменную
            X = data[feature_columns].values
            y = data[target_column].values

            # Обучение модели LogReg
            logreg_model = LogReg(n_inputs=X.shape[1])
            logreg_model.fit(X, y)

            # Обучение модели sklearn
            sklearn_model = LogisticRegression()
            sklearn_model.fit(X, y)

            # Получение предсказаний
            logreg_predictions = logreg_model.predict(X)
            sklearn_predictions = sklearn_model.predict(X)

            # Визуализация результатов
            st.subheader("Сравнение предсказаний")
            plt.figure(figsize=(10, 6))

            #plt.scatter(data[feature_columns[0]], data[feature_columns[1]], c=y, cmap='bwr', alpha=0.5, label='Выдача кредитов')
            plt.scatter(data[feature_columns[0]][y ==1], data[feature_columns[1]][y ==1], color='blue', label='Кредит выдан')
            plt.scatter(data[feature_columns[0]][y ==0], data[feature_columns[1]][y ==0], color='red', label='Кредит не выдан')

            # Линия для LogReg
            x_min, x_max = data[feature_columns[0]].min(), data[feature_columns[0]].max()
            y_min = -(logreg_model.intercept_ + logreg_model.coef_[0] * x_min) / logreg_model.coef_[1]
            y_max = -(logreg_model.intercept_ + logreg_model.coef_[0] * x_max) / logreg_model.coef_[1]
            plt.plot([x_min, x_max], [y_min, y_max], label='Созданный класс LogReg', color='orange')

            # Линия для sklearn
            y_min_sklearn = -(sklearn_model.intercept_ + sklearn_model.coef_[0][0] * x_min) / sklearn_model.coef_[0][1]
            y_max_sklearn = -(sklearn_model.intercept_ + sklearn_model.coef_[0][0] * x_max) / sklearn_model.coef_[0][1]
            plt.plot([x_min, x_max], [y_min_sklearn, y_max_sklearn], label='Модель Sklearn', color='blue')

            plt.xlabel(feature_columns[0])
            plt.ylabel(feature_columns[1])
            plt.title("Сравнение моделей логистической регрессии")
            plt.legend()
            st.pyplot(plt)

            logreg_score = round(accuracy_score(y, logreg_model.predict(X)),3)
            sklearn_score = round(accuracy_score(y, sklearn_model.predict(X)),3)
                  
            st.write(f"Точность класса LogReg: {round(accuracy_score(y, logreg_model.predict(X)),3)}")
            st.write(f"Точность модели sklearn: {round(accuracy_score(y, sklearn_model.predict(X)),3)}")