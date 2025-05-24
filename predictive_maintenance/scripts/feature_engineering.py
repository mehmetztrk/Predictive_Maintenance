import pandas as pd

def preprocess_data(input_path, output_path):
    # Veriyi oku
    df = pd.read_csv(input_path)

    # 1. Yeni özellik: Mekanik güç
    df['mechanical_power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

    # 2. Kategorik değişkenleri label encode et
    df['Type'] = df['Type'].astype('category').cat.codes

    # 3. Gereksiz sütunları kaldır (ID gibi)
    df.drop(columns=['UDI', 'Product ID'], inplace=True, errors='ignore')

    # 4. NaN varsa doldur (gerekiyorsa)
    df.fillna(0, inplace=True)

    # 5. Hazırlanan veriyi kaydet
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_data("data/ai4i2020.csv", "data/processed_data.csv")
