import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler



model = keras.Sequential(
    [
        layers.Dense(5, activation="relu", name="l1"),
        layers.Dense(12, name="l2"),
    ]
 )

train_samples = []
train_labels = []

for i in range(50):
    young = randint(13, 64)
    train_samples.append(young)
    train_labels.append(1)
    old = randint(65, 100)
    train_samples.append(old)
    train_labels.append(0)

for i in range(1000):
    young = randint(13, 64)
    train_samples.append(young)
    train_labels.append(0)
    old = randint(65, 100)
    train_samples.append(old)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.Array(train_samples)
mapped_samples = zip(train_samples, train_labels)
random.shuffle(mapped_samples)
train_samples, train_labels = zip(*mapped_samples)

model.add(layers.Activation("softmax"))
model.compile(Adam(learning_rate=.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_samples, train_labels, batch="10", epochs="20", shuffle=True, verbose=2)


x = tf.ones((3, 3))
y = model(x)
