import numpy as np
from flask import Flask, request, jsonify
from simple_nn import SimpleNN

app = Flask(__name__)

nn = SimpleNN(784, 50, 10, load_model=True)

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image_data']
    image_data = np.array(data).flatten() / 255.0
    image_data = np.expand_dims(image_data, axis=0)
    prediction = nn.forward(image_data)
    return jsonify({'prediction': int(np.argmax(prediction)), 'probabilities': prediction.tolist()[0]})

if __name__ == '__main__':
    app.run(debug=True)
