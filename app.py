import os
from flask import Flask, request, render_template
import pickle
import numpy as np

# Change the working directory to where app.py and the model are located
os.chdir('c:/Users/richa/Desktop/project/')

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

    output = prediction[0]
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True,port=8080)
