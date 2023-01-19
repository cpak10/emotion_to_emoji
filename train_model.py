import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

# set up os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

# set up file root
file_root = "C:\\GitHub\\emotion_to_emoji"

# set up location of photos
data_dir = f"{file_root}\\intake\\training"

# create a dataset object for training and val with the images in the data_dir directory
image_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "training", seed = 123, image_size = (224, 224), color_mode = "grayscale"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "validation", seed = 123, image_size = (224, 224), color_mode = "grayscale"
)

# plot the labels for show
class_names = image_ds.class_names
plt.figure(figsize = (18, 10))
for images, labels in image_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap = "gray")
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.show()

# augment data
data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip(
        "horizontal",
        input_shape = (
            224, 224, 1
        )
    ),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
  ]
)

# set up the model
model = keras.Sequential([
    keras.layers.Rescaling(1./255, input_shape = (224, 224, 1)),
    data_augmentation,
    keras.layers.Conv2D(16, 3, padding = "same", activation = "relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding = "same", activation = "relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding = "same", activation = "relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(0.01), bias_regularizer = keras.regularizers.l2(0.01), activation = "relu"),
    keras.layers.Dense(8, activation = "softmax")
])

# compile the model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# train the model
es_cb = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5)
model.fit(image_ds, epochs = 100, validation_data = val_ds, callbacks = [es_cb])

# evaluate the model
test_loss, test_acc = model.evaluate(val_ds)
print(f"\nNOTE: Test accuracy: {int(test_acc * 100)}%")

# save model
model.save(f"{file_root}\\working\\model_sequential")