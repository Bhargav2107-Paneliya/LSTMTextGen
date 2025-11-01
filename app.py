from keras.models import load_model
import pickle
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model with optimizations
model = load_model('model1.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load max_len
with open('config.json') as f:
    config = json.load(f)
max_len = config['max_len']

# Cache for word index mapping
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

def predict_next_words(text, n_words=3):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])[0]
    
    # Pad the sequence
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence
    else:
        sequence = sequence[-max_len:]
    
    # Make prediction with optimized settings
    pred = model.predict(np.array([sequence]), verbose=0, batch_size=1)
    
    # Get top 3 predictions
    top_indices = np.argsort(pred[0])[-n_words:][::-1]
    
    # Convert predictions back to words using cached mapping
    suggestions = [index_word.get(idx, '') for idx in top_indices if idx in index_word]
    return suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    suggestions = predict_next_words(text)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
