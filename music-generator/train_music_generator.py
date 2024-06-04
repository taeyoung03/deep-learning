import tensorflow as tf
import numpy as np

text = open('pianoabc.txt', 'r').read()

unique_text = list(set(text))
unique_text.sort()

#utilities
text_to_num = {}
num_to_text = {}

for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

digital_text = []
for i in text:
    digital_text.append(text_to_num[i])

x = []
y = []

for i in range(0, len(digital_text) - 25):
    x.append(digital_text[i : i + 25])
    y.append(digital_text[i + 25])

x = tf.one_hot(x, 31)
y = tf.one_hot(y, 31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, 31)),
    tf.keras.layers.Dense(31, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x, y, batch_size=64, epochs=50)

model.save('models/model1')