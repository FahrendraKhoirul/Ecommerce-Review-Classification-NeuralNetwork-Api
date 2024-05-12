from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pipeline_nn_ecommerce_final as my_pipeline
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')
    # return "Welcome to E-commerce Review Classification API, created by Fahrendra Khoirul Ihtada. \nThis version is for testing the predict function only. \nPlease use /predict endpoint to use the model. \nThank you!"


@app.route('/predict', methods=['GET'])
def predict():
    data = request.args.get('sentence')
    result = my_pipeline.pipeline_nn_ecommerce(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)