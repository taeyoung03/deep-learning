import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=(150, 150),
    batch_size=64,
    subset="training",
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=(150, 150),
    batch_size=64,
    subset="validation",
    validation_split=0.2,
    seed=1234
)

def preprocess(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label 

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

inception_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
inception_model.load_weights("inception_v3.h5")

for layer in inception_model.layers:
    layer.trainable = False

unfreeze = False
for layer in inception_model.layers:
    if layer.name == "mixed6":
        unfreeze = True
    if unfreeze == True:
        layer.trainable =True

last_layer_output = inception_model.get_layer("mixed7").output
x = tf.keras.layers.Flatten()(last_layer_output)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inception_model.input, output)

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-6), metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)