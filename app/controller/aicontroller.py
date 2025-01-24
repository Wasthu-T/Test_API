import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from flask import request
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from app import response, app
import datetime

def index():
    df = pd.read_csv('app/controller/dataset/toyota.csv')
    return df

def check():
    df = index()
    missing = df.isnull().sum()
    info = df.info()
    shape = df.shape
    print(missing)
    print(info)
    print(shape)

def main():
    df = index()
    nilai_tukar = 16216
    df['price'] = df['price'] * nilai_tukar
    df['tax'] = df['tax'] * nilai_tukar

    # Dictionary untuk menyimpan LabelEncoder
    encoders = {}
    standards = {}
    # Label Encoding untuk setiap kolom
    for col in ['model', 'transmission', 'fuelType']:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder 
    encoders['updated_at'] = datetime.datetime.now()
    for col in ['mileage', 'tax', 'price']:
        standard_scaler = StandardScaler()
        df[col] = standard_scaler.fit_transform(df[[col]])
        standards[col] = standard_scaler
    standards['updated_at'] = datetime.datetime.now()
    joblib.dump(encoders, "app/controller/dataset/encoders.joblib")
    joblib.dump(standards, "app/controller/dataset/standards.joblib")

    
    # preporation
    features = ['model','year','transmission','mileage','fuelType','tax', 'mpg','engineSize']
    x = df[features]
    y = df['price']


    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    # model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    y_test_original = standards['price'].inverse_transform(y_test.values.reshape(-1,1))
    pred_original = standards['price'].inverse_transform(pred.reshape(-1,1))
    mse = mean_squared_error(y_test_original, pred_original)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_original, pred_original) *100
    mae = mean_absolute_error(y_test_original, pred_original)
    r2 = r2_score(y_test_original, pred_original)
    score = lr.score(X_test, y_test)
    data = {
        'lr': lr,
        'mse': mse,
        'mae' : mae,
        'rmse':rmse,
        'mape':mape,
        'r2':r2,
        'score':score,
        'updated_at': datetime.datetime.now()
    }
    joblib.dump(data, "app/controller/dataset/linear_regression_model.joblib")
    print(f"'Model saved to 'linear_regression_model.joblib'")
    return response.ok('Success', "Berhasil update model")

def validate_input(data):
    errors = []

    # Validasi tipe data dan nilai
    if not isinstance(data["year"], int) or not (1900 <= data["year"] <= 2100):
        errors.append({"field": "year", "error": "Year harus berupa integer dalam rentang 1900-2100."})
    if data["transmission"] not in ["Manual", "Automatic", "Semi-Auto"]:
        errors.append({"field": "transmission", "error": "Transmission harus berupa salah satu dari: Manual, Automatic, Semi-Auto."})
    if not isinstance(data["mileage"], (int, float)) or data["mileage"] < 0:
        errors.append({"field": "mileage", "error": "Mileage harus berupa angka non-negatif."})
    if data["fuelType"] not in ["Petrol", "Diesel", "Electric", "Hybrid"]:
        errors.append({"field": "fuelType", "error": "FuelType harus berupa salah satu dari: Petrol, Diesel, Electric, Hybrid."})
    if not isinstance(data["tax"], (int, float)) or data["tax"] < 0:
        errors.append({"field": "tax", "error": "Tax harus berupa angka non-negatif."})
    if not isinstance(data["mpg"], (int, float)) or data["mpg"] < 0:
        errors.append({"field": "mpg", "error": "MPG harus berupa angka non-negatif."})
    if not isinstance(data["engineSize"], (int, float)) or data["engineSize"] <= 0:
        errors.append({"field": "engineSize", "error": "EngineSize harus berupa angka positif."})

    return errors

def save_pred(data):
    try:
        df_old = pd.read_csv('"app/controller/dataset/pred.csv')
    except FileNotFoundError:
        df_old = pd.DataFrame()

    # Gabungkan data lama dengan data baru
    df_combined = pd.concat([df_old, data], ignore_index=True)
    df_combined.to_csv('pred.csv', index=False)
    return response.ok('Success', "Data berhasil disimpan.")

def predict():
    model_car = request.json['model']
    year = request.json['year']
    transmission = request.json['transmission']
    mileage = request.json['mileage']
    fueltype = request.json['fueltype']
    tax = request.json['tax']
    mpg = request.json['mpg']
    engineSize = request.json['enginesize']
    data = {
        "model" : model_car,
        "year" : year,
        "transmission" : transmission,
        "mileage" : mileage,
        "fuelType" : fueltype,
        "tax" : tax,
        "mpg" : mpg,
        "engineSize" : engineSize
    }
    
    validation_errors = validate_input(data)
    if validation_errors:
        return response.badRequest(validation_errors, "Validasi gagal")
    
    input_df = pd.DataFrame([data])
    # Muat model dan encoder
    model = joblib.load('app/controller/dataset/linear_regression_model.joblib')
    encoder = joblib.load('app/controller/dataset/encoders.joblib')
    scaler = joblib.load('app/controller/dataset/standards.joblib')

    # Konversi input kategorikal ke bentuk numerik
    try:
        for col in ['model', 'transmission', 'fuelType']:
            input_df[col] = encoder[col].transform(input_df[col])
    except ValueError as e:
        return response.badRequest('Fail','Terjadi kesalahan pada konversi kategorikal')

    # Standard value numerik
    try:
        for col in ['mileage', 'tax']:
            input_df[col] = scaler[col].transform(input_df[[col]])
    except ValueError as e:
        return response.badRequest('Fail','Terjadi kesalahan pada transform numerik')

    # Prediksi
    prediction = model['lr'].predict(input_df)
    prediction_original = scaler['price'].inverse_transform(prediction.reshape(-1,1))
    prediction_formatted = prediction_original[0][0]
    nilai = prediction_original[0][0]
    nilai_akhir = nilai - model['rmse']
    prediksi_harga = f"Nilai: Rp{nilai_akhir:,.0f} - Rp{prediction_formatted:,.0f}"
    input_df['price_high'] = prediction_formatted
    input_df['price_low'] = nilai_akhir
    input_df['created_at'] = datetime.datetime.now()
    return response.ok(prediksi_harga, 'Prediksi harga berhasil')
    
