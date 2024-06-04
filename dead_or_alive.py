import tensorflow as tf
import pandas as pd

data = pd.read_csv('train.csv')

data.fillna({'Age': 30}, inplace=True)
data.fillna({'Embarked': 'S'}, inplace=True)

ans = data.pop('Survived')
ds = tf.data.Dataset.from_tensor_slices((dict(data), ans))

feature_columns = []

#numeric
feature_columns.append(tf.feature_column.numeric_column('Fare'))
feature_columns.append(tf.feature_column.numeric_column('Parch'))
feature_columns.append(tf.feature_column.numeric_column('SibSp'))

#bucketized
Age = tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10, 20, 30, 40, 50, 60])
feature_columns.append(Age_bucket)

#indicator
indicator_colums = ['Sex', 'Embarked', 'Pclass']
for col in indicator_colums:
    vocab = data[col].unique()
    cat = tf.feature_column.categorical_column_with_vocabulary_list(col, vocab)
    one_hot = tf.feature_column.indicator_column(cat)
    feature_columns.append(one_hot)

#embedding
vocab = data['Ticket'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', vocab)
one_hot = tf.feature_column.embedding_column(cat, dimension=9)
feature_columns.append(one_hot)

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

ds_batch = ds.batch(32)

model.fit(ds_batch, shuffle=True, epochs=30)