from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

model = tf.keras.saving.load_model("multiModelNetwork.keras")

bgrImg = cv2.imread("resizedDataset - Kopi/Scene1/00022_00193_outdoor_000_0000.png")
img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)

for i, layer in enumerate(model.layers):
    if "conv" not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(i, layer.name, layer.output.shape)

# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
 # get the filter
 f = filters[:, :, :, i]
 # plot each channel separately
 for j in range(3):
     # specify subplot and turn of axis
     ax = pyplot.subplot(n_filters, 3, ix)
     ax.set_xticks([])
     ax.set_yticks([])
     # plot filter channel in grayscale
     pyplot.imshow(f[:, :, j], cmap='gray')
     ix += 1
# show the figure
pyplot.show()

ixs = [1, 3, 5, 7]
outputs = [model.layers[i+1].output for i in ixs]
model = keras.Model(inputs=model.inputs, outputs=outputs)
model.summary()

feature_maps = model.predict(img)
# plot the output from each block
square = 8
mapdepths = [1, 2, 4, 8]
mapindex = 0
for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(mapdepths[mapindex]):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    mapindex += 1

'''
model = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
feature_maps = model.predict(img)
ix = 1
for _ in range(1):
    for _ in range(8):
        # specify subplot and turn of axis
        ax = pyplot.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
# show the figure
pyplot.show()


model = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
feature_maps = model.predict(img)
ix = 1
for _ in range(2):
    for _ in range(8):
        # specify subplot and turn of axis
        ax = pyplot.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
# show the figure
pyplot.show()

model = keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
feature_maps = model.predict(img)
ix = 1
for _ in range(4):
    for _ in range(8):
        # specify subplot and turn of axis
        ax = pyplot.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
# show the figure
pyplot.show()

model = keras.Model(inputs=model.inputs, outputs=model.layers[7].output)
feature_maps = model.predict(img)
ix = 1
for _ in range(8):
    for _ in range(8):
        # specify subplot and turn of axis
        ax = pyplot.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
show the figure
'''