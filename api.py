from flask import Flask, request, jsonify
import preprocessing as pre
import pickle

app = Flask(__name__)
tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def index():
    return "Welcome to E-commerce Review Classification API, created by Fahrendra Khoirul Ihtada. \nThis version is for testing the predict function only. \nPlease use /predict endpoint to use the model. \nThank you!"

@app.route('/predict', methods=['GET'])
def predict():
    data = request.args.get('sentence')
    sentence = data
    preprocess_output = pre.preprocess(sentence)
    tfidf_vector = tfidf_model.transform(preprocess_output)

    #  make result that contain sentence, preprocess_output, and tfidf_vector
    result = {
        "success": True,
        "data": {
        'sentence': sentence,
        'preprocess': preprocess_output,
        'tfidf': tfidf_vector.tolist()
        }
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',port=10000)