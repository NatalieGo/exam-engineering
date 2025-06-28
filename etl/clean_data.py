import argparse
import pandas as pd

def clean_data(input_path, output_path):
    """
    Загружает данные из csv-файла, преобразует diagnosis в числовой формат,
    удаляет столбец с id, проводит проверку на пропуски, сохраняет очищенный датасет.
    """
    df = pd.read_csv(input_path)

    # Преобразуем diagnosis: M = 1 (Malignant - злокачественная), B = 0 (Benign - доброкачественная)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Удаляем столбец id
    df.drop(columns=['id'], inplace=True)

    # Проверяем пропущенные значения и при наличии удаляем
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Обнаружено пропущенных значений: {missing}, удаляем строки с пропусками")
        df.dropna(inplace=True)

    # Сохраняем очищенный датасет
    df.to_csv(output_path, index=False)
    print(f"Очистка завершена. Сохранено в: {output_path}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Очистка данных Breast Cancer Wisconsin Diagnostic")
    parser.add_argument('--input_path', type=str, default='breast_cancer_raw.csv', help="Путь к исходному CSV с сырыми данными")
    parser.add_argument('--output_path', type=str, default='breast_cancer_clean.csv', help="Путь для сохранения очищенных данных")
    args = parser.parse_args()

    clean_data(args.input_path, args.output_path)
