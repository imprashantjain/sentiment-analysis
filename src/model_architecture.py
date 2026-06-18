from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

model = Sequential([
    Input(shape=(250,)),

    Embedding(
        input_dim=10000,
        output_dim=16
    ),

    LSTM(32),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.summary()