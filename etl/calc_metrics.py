import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import os

def calc_metrics(input_path, model_path, output_path):
    """
    Загружает очищенные данные и обученную модель,
    предсказывает на тестовой выборке и рассчитывает метрики,
    сохраняет результаты в json-файл.
    """
    # Загружаем данные
    df = pd.read_csv(input_path)

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Разбиваем на train/test (такие же параметры, как при обучении)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Загружаем сохранённый pipeline
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Делаем предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    # РАссчитываем метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Создаём папку для сохранения
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Сохраняем метрики в json-файл
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Метрики сохранены в: {output_path}")
    print(metrics)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расчет метрик модели")
    parser.add_argument('--input_path', type=str, default='breast_cancer_clean.csv', help="Путь к очищенным данным")
    parser.add_argument('--model_path', type=str, default='model.pkl', help="Путь к обученной модели")
    parser.add_argument('--output_path', type=str, default='results/metrics.json', help="Путь для сохранения метрик")
    args = parser.parse_args()

    calc_metrics(args.input_path, args.model_path, args.output_path)
