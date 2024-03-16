from flask import Flask, request, jsonify
from flask_cors import CORS
import pipeline_nn_ecommerce as my_pipeline

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return "Welcome to E-commerce Review Classification API, created by Fahrendra Khoirul Ihtada. \nThis version is for testing the predict function only. \nPlease use /predict endpoint to use the model. \nThank you!"

# @app.route('/predict', methods=['GET'])
# def predict():
#     data = request.args.get('sentence')
#     sentence = data
#     preprocess_output = pre.preprocess(sentence)
#     tfidf_vector = tfidf_model.transform(preprocess_output)
#     top_words = tfidf_model.getTopWord(tfidf_vector, 5)

#     #  make result that contain sentence, preprocess_output, and tfidf_vector
#     result = {
#         "success": True,
#         "data": {
#         'sentence': sentence,
#         'preprocess': preprocess_output,
#         'tfidf': tfidf_vector.tolist(),
#         'top_words': top_words
#         }
#     }

#     return jsonify(result)

@app.route('/predict', methods=['GET'])
def predict():
    data = request.args.get('sentence')
    result = my_pipeline.pipeline_nn_ecommerce(80, data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)