import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Training data
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]],
    [[5], [6], [7]]
])

y = np.array([4, 5, 6, 7, 8])

# Create LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Prediction
test = np.array([[[6], [7], [8]]])
prediction = model.predict(test)

print("Next predicted number:", prediction[0][0])