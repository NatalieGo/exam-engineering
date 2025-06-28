import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def train_model(input_path='breast_cancer_clean.csv', model_path='model.pkl'):
    df = pd.read_csv(input_path)

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # создаем pipeline: StandardScaler + LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Модель с нормализацией обучена и сохранена в: {model_path}")
    return pipeline

if __name__ == "__main__":
    train_model()