import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


model = keras.Sequential(
    [
        layers.Dense(5, activation="relu", name="l1"),
        layers.Dense(12, name="l2"),
    ]
 )

model.add(layers.Activation("softmax"))
model.compile(Adam(learning_rate=.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_samples, train_labels, batch="10", epochs="20", shuffle=True, verbose=2)

x = tf.ones((3, 3))
y = model(x)
