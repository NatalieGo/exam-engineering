import pandas as pd

def clean_data(input_path='breast_cancer_raw.csv', output_path='breast_cancer_clean.csv'):
    df = pd.read_csv(input_path)

    # преобразуем diagnosis: M = 1 (Malignant - злокачественная), B = 0 (Benign - доброкачественная)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # удаляем id
    df.drop(columns=['id'], inplace=True)

    # проверяем на пропущенные значения
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Обнаружено пропущенных значений: {missing}, удаляем строки с пропусками")
        df.dropna(inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Очистка завершена. Сохранено в: {output_path}")
    return df

if __name__ == "__main__":
    clean_data()