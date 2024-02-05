from flask import Flask, request, jsonify
import preprocessing as pre
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

try:
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
except FileNotFoundError:
    print("The file 'tfidf_model.pkl' was not found.")
except pickle.UnpicklingError:
    print("Could not unpickle the object. The file might be corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


@app.route('/', methods=['GET'])
def index():
    return "Welcome to E-commerce Review Classification API, created by Fahrendra Khoirul Ihtada. \nThis version is for testing the predict function only. \nPlease use /predict endpoint to use the model. \nThank you!"

@app.route('/predict', methods=['GET'])
def predict():
    data = request.args.get('sentence')
    sentence = data
    preprocess_output = pre.preprocess(sentence)
    tfidf_vector = tfidf_model.transform(preprocess_output)
    top_words = tfidf_model.getTopWord(tfidf_vector, 5)

    #  make result that contain sentence, preprocess_output, and tfidf_vector
    result = {
        "success": True,
        "data": {
        'sentence': sentence,
        'preprocess': preprocess_output,
        'tfidf': tfidf_vector.tolist(),
        'top_words': top_words
        }
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)