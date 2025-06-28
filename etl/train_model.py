import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def train_model(input_path, model_path):
    """
    Загружает очищенные данные, делит на train/test,
    обучает логистическую регрессию с нормализацией,
    сохраняет pipeline в файл.
    """
    df = pd.read_csv(input_path)

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Создаём pipeline: нормализация + логистическая регрессия
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=2000))
    ])

    # Обучаем модель
    pipeline.fit(X_train, y_train)

    # Сохраняем модель в файл
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Модель с нормализацией обучена и сохранена в: {model_path}")
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели Logistic Regression")
    parser.add_argument('--input_path', type=str, default='breast_cancer_clean.csv', help="Путь к очищенным данным")
    parser.add_argument('--model_path', type=str, default='model.pkl', help="Путь для сохранения обученной модели")
    args = parser.parse_args()

    train_model(args.input_path, args.model_path)
