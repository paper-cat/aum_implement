import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50


def get_resnet50(input_shape, classes):
    inputs = keras.Input(shape=input_shape)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(32, 32, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )(inputs)
    rescale = layers.experimental.preprocessing.Rescaling(1. / 255)(data_augmentation)
    model = ResNet50(weights=None, input_shape=input_shape, classes=classes)
    outputs = model(rescale)

    model = keras.Model(inputs, outputs)

    return model


def resnet50_logits(input_shape, classes, weight_path):
    inputs = keras.Input(shape=input_shape)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(32, 32, 3)),
            layers.experimental.preprocessing.RandomFlip("vertical",
                                                         input_shape=(32, 32, 3)),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomZoom(0.2),
        ]
    )(inputs)
    rescale = layers.experimental.preprocessing.Rescaling(1. / 255)(data_augmentation)
    model = ResNet50(weights=None, input_shape=input_shape, classes=classes)
    outputs = model(rescale)

    model = keras.Model(inputs, outputs)

    model.load_weights(weight_path).expect_partial()
    model.layers[1].layers[-1].activation = tf.keras.activations.linear

    return model
