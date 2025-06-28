import argparse
import pandas as pd

# url с исходными данными
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

# Имена колонок (в исходном файле отсутствовали)
COLUMNS = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def load_and_preview(output_path):
    """
    Загружает данные, выводит информацию о файле: размер и распределение классов,
    сохраняет в csv по указанному пути.
    """
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)

    print("Данные загружены")
    print("Размер датасета:", df.shape)
    print("Распределение классов diagnosis:\n", df['diagnosis'].value_counts())

    df.to_csv(output_path, index=False)
    print(f"Сохранено в файл: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка данных Breast Cancer Wisconsin Diagnostic")
    parser.add_argument('--output_path', type=str, default='breast_cancer_raw.csv', help="Куда сохранить файл с данными")
    args = parser.parse_args()

    load_and_preview(args.output_path)
