import tensorflow as tf
from tensorflow.keras.layers import Embedding

embedding = Embedding(
    input_dim=10000,
    output_dim=16
)

sample = tf.constant([[1,14,22,16]])

output = embedding(sample)

print(output.shape)
print(output)