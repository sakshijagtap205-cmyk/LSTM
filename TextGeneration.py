import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Training text
text = """
deep learning is powerful
deep learning uses neural networks
lstm is used for sequence learning
neural networks learn from data
"""

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])

sequences = tokenizer.texts_to_sequences([text])[0]

X = []
y = []

for i in range(1, len(sequences)):
    X.append(sequences[:i])
    y.append(sequences[i])

X = tf.keras.preprocessing.sequence.pad_sequences(
    X, padding='pre'
)

X = np.array(X)
y = np.array(y)

# Build model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 50),
    LSTM(100),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Train
model.fit(X, y, epochs=200, verbose=1)

# Generate text
seed = "deep learning"
for i in range(5):
    seq = tokenizer.texts_to_sequences([seed])[0]
    seq = tf.keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=X.shape[1], padding='pre'
    )

    prediction = model.predict(seq, verbose=0)
    word_index = np.argmax(prediction)

    output_word = tokenizer.index_word.get(word_index, "")
    seed += " " + output_word

print("Generated Text:", seed)