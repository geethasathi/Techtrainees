from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the trained model
with open('rating_predictor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assuming the TF-IDF vectorizer and OneHotEncoder are the same as those used during training.
# These should be saved during training and loaded here. For this example, we assume they're refitted here.
# Load or refit your preprocessors (Note: In practice, use the same fitted instances from training)
# Assuming they were trained and saved before
tfidf = TfidfVectorizer(max_features=5000)
encoder = OneHotEncoder(sparse_output=False)

# Example: Load a previously fitted TF-IDF and encoder (if saved during training)
# with open('tfidf_vectorizer.pkl', 'rb') as file:
#     tfidf = pickle.load(file)
# with open('onehot_encoder.pkl', 'rb') as file:
#     encoder = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract 'description' and 'genre' from the JSON request
    description = data.get('description', '')
    genre = data.get('genre', '')

    # Transform the inputs using the same TF-IDF and OneHotEncoder as during training
    X_desc = tfidf.transform([description]).toarray()
    X_genre = encoder.transform([[genre]])

    # Combine the features
    X = np.hstack((X_desc, X_genre))

    # Make a prediction
    prediction = model.predict(X)

    # Return the prediction as a JSON response
    return jsonify({'ratingValue': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
