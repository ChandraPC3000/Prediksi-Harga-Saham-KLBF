from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load model XGBoost tanpa optimasi (Default)
model_default = xgb.Booster()
model_default.load_model("models/model_xgboost.json")  # Menggunakan file JSON

# Load model XGBoost dengan optimasi PSO
model_pso = xgb.Booster()
model_pso.load_model("models/model_xgboost_pso.json")  # Menggunakan file JSON

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Ambil input dari form HTML
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    close_price = float(request.form['close'])

    # Data input untuk model
    input_data = np.array([[open_price, high_price, low_price, close_price]])

    # Convert input ke DMatrix (format yang diterima XGBoost)
    dmatrix_input = xgb.DMatrix(input_data)

    # Prediksi menggunakan kedua model
    predicted_close_pso = model_pso.predict(dmatrix_input)[0]
    predicted_close_default = model_default.predict(dmatrix_input)[0]

    # Evaluasi metrik (contoh data y_test dan y_pred statis untuk demo)
    y_test = np.array([1500, 1520, 1535, 1550])
    y_pred_pso = np.array([1495, 1525, 1530, 1545])  # Contoh prediksi model PSO
    y_pred_default = np.array([1490, 1522, 1532, 1542])  # Contoh prediksi model default

    # Metrik untuk model PSO
    mse_pso = mean_squared_error(y_test, y_pred_pso)
    rmse_pso = np.sqrt(mse_pso)
    mae_pso = mean_absolute_error(y_test, y_pred_pso)
    mape_pso = mean_absolute_percentage_error(y_test, y_pred_pso)
    r2_pso = r2_score(y_test, y_pred_pso)

    # Metrik untuk model default
    mse_default = mean_squared_error(y_test, y_pred_default)
    rmse_default = np.sqrt(mse_default)
    mae_default = mean_absolute_error(y_test, y_pred_default)
    mape_default = mean_absolute_percentage_error(y_test, y_pred_default)
    r2_default = r2_score(y_test, y_pred_default)

    # Hasil metrik
    metrics = {
        'pso': {
            'mse': mse_pso,
            'rmse': rmse_pso,
            'mae': mae_pso,
            'mape': mape_pso,
            'r2': r2_pso,
        },
        'default': {
            'mse': mse_default,
            'rmse': rmse_default,
            'mae': mae_default,
            'mape': mape_default,
            'r2': r2_default,
        }
    }

    # Kirim hasil ke template HTML
    return render_template(
        'index.html',
        prediction_pso=f"Harga prediksi (PSO): {predicted_close_pso:.2f}",
        prediction_default=f"Harga prediksi (Default): {predicted_close_default:.2f}",
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)