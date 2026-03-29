from flask import Flask, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your ML model
model = joblib.load("stress_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict():
    hr = 80
    temp = 36.5

    x = np.array([[hr, temp,0,0,0,0]])
    prediction = model.predict(x)[0]
    # Convert to readable label
    if prediction == 0:
      level = "Low"
    elif prediction == 1:
      level = "Medium"
    else:
      level = "High"

    return jsonify({
        "heart_rate": hr,
        "temperature": temp,
        "stress_level": level
    })

if __name__ == '__main__':
    app.run(debug=True)