from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('breast_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Split the input string into individual float values
    features = [float(x) for x in request.form['features'].split(',')]
    prediction = model.predict([features])[0]
    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'
    return render_template('index.html', prediction_text=f'Predicted Cancer Type: {result}')

if __name__ == '__main__':
    app.run(debug=True)
