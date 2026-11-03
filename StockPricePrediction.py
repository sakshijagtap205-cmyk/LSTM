import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

prices = data[['Close']].values

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=10, batch_size=32)

# Prediction
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

plt.plot(prices[60:], label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("LSTM Stock Price Prediction")
plt.legend()
plt.show()