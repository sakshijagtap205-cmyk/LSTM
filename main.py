
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. Set parameters
vocab_size = 10000
max_length = 200
embedding_dim = 64

# 2. Load IMDB dataset
print("Loading IMDB dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Pad sequences
X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# 4. Build LSTM model
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length
    ),

    LSTM(
        64,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

# 5. Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6. Model summary
model.summary()

# 7. Train model
print("\nTraining LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 8. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# 9. Plot accuracy
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("LSTM Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 10. Plot loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("LSTM Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 11. Save model
model.save("lstm_sentiment_model.keras")

print("\nModel saved successfully!")

# 12. Predict custom review
word_index = imdb.get_word_index()

def predict_sentiment(review):

    words = review.lower().split()

    encoded = [1, 2]

    for word in words:
        index = word_index.get(word)

        if index is not None and index < vocab_size - 3:
            encoded.append(index + 3)
        else:
            encoded.append(2)

    padded = pad_sequences(
        [encoded],
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)[0][0]

    if prediction >= 0.5:
        print("Positive Review 😊")
    else:
        print("Negative Review 😞")

    print("Prediction Score:", prediction)


# Test predictions
predict_sentiment("This movie is amazing and wonderful")
predict_sentiment("This movie is boring and terrible")
