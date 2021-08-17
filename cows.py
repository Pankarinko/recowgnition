import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential(
    [
        layers.Dense(5, activation="relu", name="l1"),
        layers.Dense(12, name="l2"),
    ]
    )

model.add(Activation("softmax"))
