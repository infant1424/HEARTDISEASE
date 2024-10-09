import os
from flask import Flask, request, render_template
import pickle
import numpy as np

# Set the directory where your app.py and model are located (optional)
# os.chdir('/path/to/your/directory')

app = Flask(__name__)

# Load the trained model
model_path = 'decision_tree_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect features from the form
    features = [float(x) for x in request.form.values()]

    # Ensure the features are in the correct format for prediction
    final_features = [np.array(features)]

    # Perform the prediction
    prediction = model.predict(final_features)

    # 1 means heart disease is present, 0 means no heart disease
    output = "Heart disease is present" if prediction[0] == 1 else "No heart disease"

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
