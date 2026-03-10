from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# Load Model, Tokenizer, and Config
model = None
tokenizer = None
max_sequence_len = 0

def load_resources():
    global model, tokenizer, max_sequence_len
    try:
        model = tf.keras.models.load_model('model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
            max_sequence_len = config['max_sequence_len']
        print("Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}. Did you run train.py first?")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500
    
    data = request.json
    seed_text = data.get('text', '')
    
    if not seed_text:
        return jsonify({'prediction': ''})

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
            
    return jsonify({'prediction': output_word})

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_resources()
    app.run(port=5000, debug=True)
