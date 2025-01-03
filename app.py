"""### **9. GUI**

#### **9.1 Install Flask dan Ngrok**
"""
from pyngrok import ngrok

# Masukkan token Anda di sini
ngrok.set_auth_token("2qenvfvg5vXZgyTmQZbxGBx437m_3kRq219rNYWmuoBjrDpNm")

"""#### **9.2 Backend Flask**"""

from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
from pyngrok import ngrok

# Load model XGBoost PSO
model_pso = pickle.load(open("models/model_xgboost_pso.pkl", "rb"))

# Load model XGBoost tanpa PSO
model_default = pickle.load(open("models/xgboost_model_default.pkl", "rb"))

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    close_price = float(request.form['close'])

    # Predict next day close price
    input_data = [[close_price]]
    predicted_close = model.predict(input_data)[0]

    # Example evaluation metrics (static for demo)
    y_test = np.array([1500, 1520, 1535, 1550])
    y_pred = np.array([1495, 1525, 1530, 1545])
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

    return render_template(
        'index.html',
        prediction=f"Harga prediksi untuk hari berikutnya: {predicted_close:.2f}",
        metrics=metrics
    )

# Mulai ngrok untuk URL publik
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Jalankan Flask
if __name__ == "__main__":
    app.run(port=5000, debug=True)