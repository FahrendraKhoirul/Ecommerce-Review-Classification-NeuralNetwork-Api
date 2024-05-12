import preprocessing as pre
import pickle
import numpy as np


# Load TFIDF model
with open('TFIDF Model - 80 Final.pkl', 'rb') as f:
    tfidf_model_final = pickle.load(f)
    print("TFIDF Model Loaded", len(tfidf_model_final.word_list))

# load Neural Network Model
with open('NeuralNetwork_Model_B3.pkl', 'rb') as f:
    nn_model_B3 = pickle.load(f)

def pipeline_nn_ecommerce(text):

    text_clean = pre.preprocess(text)
#     print("text_clean: ", text_clean)
    text_tfidf_vector = tfidf_model_final.transform(text_clean)
    # print("text_tfidf_vector: ", text_tfidf_vector.shape)
    text_tfidf_vector_reshape = text_tfidf_vector.reshape(-1, 1) # reshape to 2D array
    # print("text_tfidf_vector: ", text_tfidf_vector.shape)
#     print(type(text_tfidf_vector))
    A0, nn_Z1, nn_A1, nn_Z2, nn_A2 = nn_model_B3.forward(text_tfidf_vector_reshape)
    # print("A0: ", A0.shape)
    # print("nn_Z1: ", nn_Z1.shape)
    # print("nn_A1: ", nn_A1.shape)
    # print("nn_Z2: ", nn_Z2.shape)
    # print("nn_A2: ", nn_A2.shape)

    nn_W1 = nn_model_B3.W1
    nn_W2 = nn_model_B3.W2
    nn_b1 = nn_model_B3.b1
    nn_b2 = nn_model_B3.b2
    prediction = np.argmax(nn_A2).item()
    print("prediction type: ", type(prediction))

    result = {
        "success": True,
        "model": f"Neural Network - B3",
        "data": {
            'sentence': text,
            'preprocess': text_clean,
            'tfidf': text_tfidf_vector.tolist(),
            'top_words': tfidf_model_final.getTopWord(text_tfidf_vector, 5),
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

def just_prediction(text):
    tfidf_model = tfidf_model_final
    nn_model = nn_model_B3

    text_clean = pre.preprocess(text)
    text_tfidf_vector = tfidf_model.transform(text_clean)
    text_tfidf_vector_reshape = text_tfidf_vector.reshape(-1, 1)
    _, _, _, _, nn_A2 = nn_model.forward(text_tfidf_vector_reshape)

    prediction = np.argmax(nn_A2).item()
    map_label = {0: 'Product', 1: 'Customer Service', 2: 'Shipping/Delivery'}
    result = map_label[prediction]
    return result

if __name__ == "__main__":
    text = "pengiriman cepat banget, bagus juga barangnya"
    result = pipeline_nn_ecommerce(text)
    print(result)