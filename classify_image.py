import cv2
from face_detection import findFace
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# file name
file_name = input("\nWhat is the file name: ")

# set file root
file_root = "C:\\GitHub\\emotion_to_emoji"

# load face finder
face_finder = findFace()
face_finder.save_face(file_path = f"{file_root}\\intake\\identify\\{file_name}.jpg", root = file_root)

# resize image
image = tf.io.read_file(f"{file_root}\\working\\cropped_face.jpg")
image = tf.image.decode_jpeg(image, channels = 1)
image = tf.image.resize(image, [224, 224])

# load model
classifier = keras.models.load_model(f"{file_root}\\working\\model_sequential")

# predict 
classes = [
    "anger", "contempt", "disgust", "fear", "happiness", "neutrality",
    "sadness", "surprise"
]
predictions = classifier.predict(image[tf.newaxis, ...])
predicted_class = tf.argmax(predictions[0])
print("\npredicted class:", classes[predicted_class], "\n")
for desc, per in zip(classes, predictions[0]):
    print(f"  {desc}: {int(per * 100)}%")