import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

raw = pd.read_table('shopping.txt', names=['rating', 'review'])
raw['label'] = np.where(raw['rating'] > 3, 1, 0)
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True)
raw.drop_duplicates(subset=['review'], inplace=True)

unique_character = raw['review'].tolist()
unique_character = ''.join(unique_character)
unique_character = list(set(unique_character))
unique_character.sort()

tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
character_list = raw['review'].tolist()
tokenizer.fit_on_texts(character_list)

train_seq = tokenizer.texts_to_sequences(character_list)
x = pad_sequences(train_seq, maxlen=100)

y = raw['label'].tolist()

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_x = np.array(train_x)
val_x = np.array(val_x)
train_y = np.array(train_y)
val_y = np.array(val_y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(unique_character) + 1, 16),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=64, epochs=10)