from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and scaler using absolute paths
model_path = os.path.join(current_dir, 'model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

print(f"Looking for model at: {model_path}")
print(f"Looking for scaler at: {scaler_path}")

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [
            float(request.form['year']),
            float(request.form['hour']),
            float(request.form['cost'])
        ]

        # Scale the features
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Get prediction probabilities
        proba = model.predict_proba(features_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        return render_template('index.html',
                               prediction_text=f'Bike Status Prediction: {prediction}',
                               confidence_text=f'Confidence: {confidence}%',
                               input_text=f'Input: Year={features[0]}, Hour={features[1]}, Cost=${features[2]}')

    except Exception as e:
        return render_template('index.html',
                               error=f'Error making prediction: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)