import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

images = []
file_list = os.listdir('archive/img_align_celeba/img_align_celeba')
for i in file_list[0 : 50000]:
    digital_image = Image.open('archive/img_align_celeba/img_align_celeba/' + i).crop((20, 30, 160, 180)).convert('L').resize((64, 64))
    images.append(np.array(digital_image))
images = np.array(images)
images = np.divide(images, 255)
images = images.reshape(50000, 64, 64, 1)

#Discriminator
discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[64, 64, 1]),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

#Generator
noise_shape = 100
generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(noise_shape,) ), 
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])
generator.summary()

GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False
GAN.compile(optimizer='adam', loss='binary_crossentropy')

def predict_pic():
    predict = generator.predict(np.random.uniform(-1, 1, size=(10, 100)))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(predict[i].reshape(64, 64), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

correct = np.ones(shape=(128, 1))
wrong = np.zeros(shape=(128, 1))
for i in range(100):
    print(f'*******************epoch {i}******************')
    for j in range(50000 // 128):
        #train discriminator
        real_images = images[j * 128 : (j + 1) * 128]
        fake_images = generator.predict(np.random.uniform(-1, 1, size=(128, 100)))
        loss1 = discriminator.train_on_batch(real_images, correct)
        loss2 = discriminator.train_on_batch(fake_images, wrong)
        
        #train generator
        loss3 = GAN.train_on_batch(np.random.uniform(-1, 1, size=(128, 100)), correct)

        # print(f'loss: Discriminator {loss1 + loss2} GAN {loss3}')

predict_pic()