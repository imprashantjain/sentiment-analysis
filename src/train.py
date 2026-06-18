import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Input

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data(
    num_words=10000
)

x_train=pad_sequences(x_train,maxlen=250)
x_test=pad_sequences(x_test,maxlen=250)

model=Sequential([
    Input(shape=(250,)),
    Embedding(10000,16),
    LSTM(32),
    Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)
history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)

# SAVE MODEL
model.save("sentiment_model.keras")

# EVALUATE MODEL
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")