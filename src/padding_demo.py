import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=10000)

x_train = pad_sequences(
    x_train,
    maxlen=250
)

print(x_train.shape)

print(x_train[0][:20])

print(len(x_train[0]))