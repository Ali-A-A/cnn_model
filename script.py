import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np

train_cat_dir = './cats_and_dogs_filtered/train/cats'
train_dog_dir = './cats_and_dogs_filtered/train/dogs'
test_cat_dir = './cats_and_dogs_filtered/validation/cats'
test_dog_dir = './cats_and_dogs_filtered/validation/dogs'
train_dir = './cats_and_dogs_filtered/train'
test_dir = './cats_and_dogs_filtered/validation'

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data = train_image_generator.flow_from_directory(batch_size = 10 , directory = train_dir ,
shuffle = True , target_size = (150 , 150) , class_mode = 'binary')

test_data = train_image_generator.flow_from_directory(batch_size = 10 , directory = test_dir ,
shuffle = True , target_size = (150 , 150) , class_mode = 'binary')

sample_training_images , sample_training_labels = next(train_data)



def plotImages2(images_arr , labels):
    for img,label in zip(images_arr , labels):
        plt.imshow(img)
        plt.xlabel(label)
        plt.tight_layout()
        plt.show()

# plotImages2(sample_training_images[:5] , sample_training_labels[:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# model.summary()

history = model.fit_generator(train_data , epochs=5 , validation_data=test_data , steps_per_epoch=10 , validation_steps=5)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()