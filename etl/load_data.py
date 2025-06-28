import pandas as pd

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

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

def load_and_preview(output_path='breast_cancer_raw.csv'):
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
    print("Данные загружены")
    print("Размер датасета:", df.shape)
    print("Распределение классов diagnosis:\n", df['diagnosis'].value_counts())
    df.to_csv(output_path, index=False)
    print(f"Сохранено в файл: {output_path}")

load_and_preview()