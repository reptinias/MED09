import tensorflow as tf
from tensorflow import keras
from keras import layers

def getEncoderModel():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(256, 256, 3), filters=8, kernel_size=3,
                            strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Conv1'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Conv2D(input_shape=(128, 128, 8), filters=16, kernel_size=3,
                            strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Conv2'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Conv2D(input_shape=(64, 64, 16), filters=32, kernel_size=3,
                            strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Conv3'),
        keras.layers.MaxPool2D(2, 2),

        keras.layers.Conv2D(input_shape=(32, 32, 32), filters=64, kernel_size=3,
                            strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Conv4'),
        keras.layers.MaxPool2D(2, 2),

        # flatten into 96
        keras.layers.Flatten(),


        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Dense1'),
        tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='Dense2'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Model metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def getFullNet():
    # encoding
    input = keras.Input(shape=(256,256,3), name="img1")
    x = layers.Conv2D(8, 3, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same", name='conv1')(input)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(x)

    x = layers.Conv2D(16, 3, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same", name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(x)

    x = layers.Conv2D(32, 3, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same", name='conv3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max3')(x)

    x = layers.Conv2D(64, 3, strides=1, activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding="same", name='conv4')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max4')(x)

    # Linear net
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='dense1')(x)
    x = layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='dense2')(x)
    x = layers.Dense(4, activation='softmax', name='pred')(x)

    return keras.Model(inputs=input, outputs=x, name='model_1')