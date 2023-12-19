import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np

model = tf.keras.models.load_model('MultiModelNets/MultiModelNetwork.keras')

folder = "Dataset"
filenames = []
images = []
index = 1
for filename in os.listdir(folder):
    print(index)
    img = cv2.cvtColor(cv2.imread(os.path.join(folder,filename)), cv2.COLOR_BGR2RGB)
    if img is not None:
        output = cv2.resize(img, [256, 256])
        images.append(output)
        filenames.append(filename)
    index += 1

for index, image in enumerate(images):
    #(256, 256, 3)
    prediction = model.predict(tf.convert_to_tensor(np.expand_dims(image, axis=0)))
    print(filenames[index], np.argmax(prediction, axis=1))