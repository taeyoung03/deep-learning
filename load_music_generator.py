import tensorflow as tf
import numpy as np

predict_model = tf.keras.models.load_model('models/model1')

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

current_input = digital_text[117 : 117 + 25]
current_input = tf.one_hot(current_input, 31)
current_input = tf.expand_dims(current_input, axis=0)

music = []    

for i in range(200):
    predict_val = predict_model.predict(current_input)
    predict_val = np.argmax(predict_val[0])
    music.append(predict_val)

    next_input = current_input.numpy()[0][1:]

    predict_val_one_hot = tf.one_hot(predict_val, 31)

    current_input = np.vstack([next_input, predict_val_one_hot.numpy()])
    current_input = tf.expand_dims(current_input, axis=0)

print(music)

music_text = []
for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))