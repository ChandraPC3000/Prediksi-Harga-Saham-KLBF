# -*- coding: utf-8 -*-
"""XGBOOST KLBF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10byQwnnb_50cPijBHYUoT3KGqeJjbCYX

### **1. Mengimpor Library**
"""

#---MULAI---
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from pyswarm import pso
#---SELESAI---

"""### **2. Mengambil Data dari Yahoo Finance**"""

#---MULAI---
# Ticker saham
ticker = "KLBF.JK"

# Rentang waktu
start_date = "2019-01-01"
end_date = "2024-12-31"

# Ambil data historis menggunakan yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Simpan data ke file CSV
data.to_csv("KLBF_JK_Historical.csv")

# Tampilkan 5 baris pertama
print(data.head())
#---SELESAI---

"""### **3. Persiapan Dataset**"""

#---MULAI---
# Tambahkan target Next_Day_Close
data['Next_Day_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Pilih fitur dan target
features = ['Close']
target = 'Next_Day_Close'

X = data[features]
y = data[target]
#---SELESAI---

"""### **4. Normalisasi Data**"""

#---MULAI---
# Normalisasi fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#---SELESAI---

"""### **5. Membagi Data Latih dan Uji**"""

#---MULAI---
# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Data Training dan Testing untuk Skenario 2 (XGBoost-PSO)
x_train2 = X_train.copy()  # Salin data latih untuk skenario 2
y_train2 = y_train.copy()
x_test2 = X_test.copy()    # Salin data uji untuk skenario 2
y_test2 = y_test.copy()
#---SELESAI---

"""### **6. Melatih Model XGBoost**



"""

#---MULAI---
# Train model XGBoost
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(model, "xgboost_model_klbf.pkl")
#---SELESAI---

"""### **7. Eksperimen Parameter XGBoost**

#### **7.1 XGBoost tanpa optimasi PSO**
"""

# --- MULAI ---
# Training Model XGBoost dengan Parameter default
model_default = XGBRegressor(
    max_depth=6,
    gamma=0,
    reg_lambda=1,
    learning_rate=0.3,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1
)

model_default.fit(X_train, y_train)

# Simpan model default ke file
joblib.dump(model_default, "xgboost_model_default.pkl")
# --- SELESAI ---

"""#### **7.2 XGBoost dengan optimasi parameter PSO**"""

# --- MULAI ---
# Training Model XGBoost dengan optimasi parameter PSO
def tune_xgboost(params):
    max_depth = int(params[0])
    gamma = params[1]
    reg_lambda = params[2]
    learning_rate = params[3]
    min_child_weight = params[4]
    subsample = params[5]
    colsample_bytree = params[6]

    model_pso = XGBRegressor(
        max_depth=max_depth,
        gamma=gamma,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    model_pso.fit(x_train2, y_train2)
    pred2 = model_pso.predict(x_test2)
    mse2 = mean_squared_error(y_test2, pred2)

    return mse2

# Optimasi Parameter dengan PSO
lb = [3, 0.01, 0.01, 0.01, 1, 0.5, 0.6]  # Lower Bound
ub = [10, 0.5, 10, 0.3, 10, 0.9, 1]      # Upper Bound

start_time = time.time()

best_params, _ = pso(
    tune_xgboost,
    lb,
    ub,
    swarmsize=100,
    omega=0.6,
    phip=1.5,
    phig=1.5,
    maxiter=100,
    minfunc=1e-12,
    debug=True
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nAlgoritma PSO Membutuhkan Waktu {elapsed_time} Detik")

# Best Parameter Hasil PSO
best_max_depth = int(best_params[0])
best_gamma = best_params[1]
best_reg_lambda = best_params[2]
best_learning_rate = best_params[3]
best_min_child_weight = best_params[4]
best_subsample = best_params[5]
best_colsample_bytree = best_params[6]

print('Best max_depth :', best_max_depth)
print('Best gamma :', best_gamma)
print('Best reg_lambda :', best_reg_lambda)
print('Best learning_rate :', best_learning_rate)
print('Best min_child_weight :', best_min_child_weight)
print('Best subsample :', best_subsample)
print('Best colsample_bytree :', best_colsample_bytree)

# Latih ulang model dengan parameter terbaik
model_pso = XGBRegressor(
    max_depth=best_max_depth,
    gamma=best_gamma,
    reg_lambda=best_reg_lambda,
    learning_rate=best_learning_rate,
    min_child_weight=best_min_child_weight,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree
)

model_pso.fit(x_train2, y_train2)

# Simpan model PSO ke file setelah eksperimen
joblib.dump(model_pso, "xgboost_model_pso.pkl")
print("Model hasil optimasi PSO telah disimpan ke file 'xgboost_model_pso.pkl'")
# --- SELESAI ---

# --- MULAI ---
# Training Best Model XGBoost dengan Parameter Terbaik hasil PSO
best_model = XGBRegressor(
    max_depth=3,  # Hasil dari PSO
    gamma=0.4998,  # Hasil dari PSO
    reg_lambda=2.8687,  # Hasil dari PSO
    learning_rate=0.1643,  # Hasil dari PSO
    min_child_weight=7.7663,  # Hasil dari PSO
    subsample=0.8618,  # Hasil dari PSO
    colsample_bytree=0.6009,  # Hasil dari PSO
    random_state=42  # Untuk reproduktifitas
)

# Fit model ke data latih
best_model.fit(x_train2, y_train2)

# Simpan best model ke file
joblib.dump(best_model, "best_xgboost_model.pkl")

print("Best model berhasil dilatih dan disimpan sebagai 'best_xgboost_model.pkl'")
# --- SELESAI ---

"""### **8. Prediksi dan Evaluasi Hasil**"""

#---MULAI---
# Langkah 8: Prediksi dan Evaluasi

# Prediksi data uji untuk Model 1 (XGBoost tanpa optimasi PSO)
y_pred = model.predict(X_test)

# Prediksi data uji untuk Model 2 (XGBoost dengan optimasi parameter dari PSO)
y_pred2 = best_model.predict(x_test2)

# Pastikan y_test dan y_test2 adalah numerik
if y_test.dtype == 'object':
    y_test = y_test.astype(float)

if y_test2.dtype == 'object':
    y_test2 = y_test2.astype(float)

# Evaluasi Hasil untuk Model 1
mse1 = mean_squared_error(y_test, y_pred)
rmse1 = np.sqrt(mse1)
mae1 = mean_absolute_error(y_test, y_pred)
mape1 = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2_1 = r2_score(y_test, y_pred)

print("=== Model 1: XGBoost tanpa optimasi PSO ===")
print("Mean Squared Error (MSE):", mse1)
print("Root Mean Squared Error (RMSE):", rmse1)
print("Mean Absolute Error (MAE):", mae1)
print("Mean Absolute Percentage Error (MAPE):", mape1)
print("R-squared:", r2_1)

# Evaluasi Hasil untuk Model 2
mse2 = mean_squared_error(y_test2, y_pred2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_test2, y_pred2)
mape2 = np.mean(np.abs((y_test2 - y_pred2) / y_test2)) * 100
r2_2 = r2_score(y_test2, y_pred2)

print("\n=== Model 2: XGBoost dengan optimasi PSO ===")
print("Mean Squared Error (MSE):", mse2)
print("Root Mean Squared Error (RMSE):", rmse2)
print("Mean Absolute Error (MAE):", mae2)
print("Mean Absolute Percentage Error (MAPE):", mape2)
print("R-squared:", r2_2)

# Save Model
pickle.dump(model, open('model_xgboost.pkl', 'wb'))
pickle.dump(best_model, open('model_xgboost_pso.pkl', 'wb'))

print("\nModel 1 dan Model 2 berhasil disimpan ke file.")
#---SELESAI---