import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import pickle
import os

# 1. Sample Corpus (You can replace this with a larger text file)
corpus = """
Next word prediction is a common task in natural language processing.
The goal is to predict the most likely next word given a sequence of words.
Bidirectional RNNs are powerful because they look at context from both directions.
LSTMs and GRUs are types of RNNs that handle long-term dependencies well.
This project uses a simple LSTM based bidirectional RNN model.
Machine learning models like this can be used for text completion and autocomplete.
The weather is very nice today.
I love programming in Python and building AI models.
Deep learning has revolutionized the field of NLP.
Artificial intelligence is the future of technology.
"""

def train_model():
    # 2. Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    total_words = len(tokenizer.word_index) + 1

    # 3. Create Input Sequences
    input_sequences = []
    for line in corpus.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # 4. Pad Sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # 5. Create Predictors and Label
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # 6. Build the Model
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        Bidirectional(LSTM(150)),
        Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Starting training...")
    model.fit(X, y, epochs=100, verbose=1)

    # 7. Save Model and Tokenizer
    model.save('model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save max_sequence_len for inference
    with open('config.pkl', 'wb') as f:
        pickle.dump({'max_sequence_len': max_sequence_len}, f)

    print("Training complete. Model and Tokenizer saved.")

if __name__ == "__main__":
    train_model()
