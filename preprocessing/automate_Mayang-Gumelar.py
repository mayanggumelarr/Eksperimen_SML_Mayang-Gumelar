from joblib import dump
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def preproces_auto(data, target_column, scaler_save_path, output_prefix):

    print("Preprocessing start...")
    # 1. Pisahkan X dan y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scalling
    scaler = RobustScaler()

    # 4. Fit dan transform ke data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Konversi kembali ke DataFrame dengan kolom n index benar
    # Train
    X_train_final = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    # Test
    X_test_final = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    # 6. Concat X dan y
    combined_train = pd.concat([X_train_final, y_train], axis=1)
    train_file_name = f'{output_prefix}_train.csv'
    combined_train.to_csv(train_file_name, index=False)

    combined_test = pd.concat([X_test_final, y_test], axis=1)
    test_file_name = f'{output_prefix}_test.csv'
    combined_test.to_csv(test_file_name, index=False)

    # 7. Save scaler
    dump(scaler, scaler_save_path)
    print("Preprocessing done!")

    return X_train, X_test, y_train, y_test

# Penggunaan
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_NAME = 'Data_Liver_Cleaned.csv'
data_path = os.path.join(SCRIPT_DIR, DATA_FILE_NAME)

df = pd.read_csv(data_path)

X_train, X_test, y_train, y_test = preproces_auto(
    data=df,
    target_column='Target',
    scaler_save_path='RobustScaler.joblib',
    output_prefix='Liver'
)