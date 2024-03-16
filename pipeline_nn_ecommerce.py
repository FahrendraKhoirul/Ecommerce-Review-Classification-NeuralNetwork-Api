import preprocessing as pre
import pickle
import numpy as np


# Load TFIDF model
with open('Model/TFIDF Model - 90.pkl', 'rb') as f:
    tfidf_model_90 = pickle.load(f)

with open('Model/TFIDF Model - 80.pkl', 'rb') as f:
    tfidf_model_80 = pickle.load(f)

with open('Model/TFIDF Model - 70.pkl', 'rb') as f:
    tfidf_model_70 = pickle.load(f)

# load Neural Network Model
with open('Model/NeuralNetwork_model_90', 'rb') as f:
    nn_model_90 = pickle.load(f)

with open('Model/NeuralNetwork_model_80', 'rb') as f:
    nn_model_80 = pickle.load(f)

with open('Model/NeuralNetwork_model_70', 'rb') as f:
    nn_model_70 = pickle.load(f)

def pipeline_nn_ecommerce(model, text):
    if model == 90:
        tfidf_model = tfidf_model_90
        nn_model = nn_model_90
    elif model == 80:
        tfidf_model = tfidf_model_80
        nn_model = nn_model_80
    elif model == 70:
        tfidf_model = tfidf_model_70
        nn_model = nn_model_70
    else:
        return {
            "success": False,
            "message": "Model not found",
            "data": {}
        }

    text_clean = pre.preprocess(text)
#     print("text_clean: ", text_clean)
    text_tfidf_vector = tfidf_model.transform(text_clean)
    # print("text_tfidf_vector: ", text_tfidf_vector.shape)
    text_tfidf_vector_reshape = text_tfidf_vector.reshape(-1, 1) # reshape to 2D array
    # print("text_tfidf_vector: ", text_tfidf_vector.shape)
#     print(type(text_tfidf_vector))
    A0, nn_Z1, nn_A1, nn_Z2, nn_A2 = nn_model.forward(text_tfidf_vector_reshape)
    # print("A0: ", A0.shape)
    # print("nn_Z1: ", nn_Z1.shape)
    # print("nn_A1: ", nn_A1.shape)
    # print("nn_Z2: ", nn_Z2.shape)
    # print("nn_A2: ", nn_A2.shape)

    nn_W1 = nn_model.W1
    nn_W2 = nn_model.W2
    nn_b1 = nn_model.b1
    nn_b2 = nn_model.b2
    prediction = np.argmax(nn_A2).item()
    print("prediction type: ", type(prediction))

    result = {
        "success": True,
        "model": f"Neural Network - {model}",
        "data": {
            'sentence': text,
            'preprocess': text_clean,
            'tfidf': text_tfidf_vector.tolist(),
            'top_words': tfidf_model.getTopWord(text_tfidf_vector, 5),
            'nn_output': {
                'A0': A0.tolist(),
                'Z1': nn_Z1.tolist(),
                'A1': nn_A1.tolist(),
                'Z2': nn_Z2.tolist(),
                'A2': nn_A2.tolist()
            },
            'nn_weight_bias': {
                'W1': nn_W1.tolist(),
                'W2': nn_W2.tolist(),
                'b1': nn_b1.tolist(),
                'b2': nn_b2.tolist()
            },
            'prediction': prediction
        }
    }
    return result

if __name__ == "__main__":
    text = "pengiriman cepat banget, bagus juga barangnya"
    model = 70
    result = pipeline_nn_ecommerce(model, text)