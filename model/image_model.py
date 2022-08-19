import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50


def get_resnet50(input_shape, classes):
    inputs = keras.Input(shape=input_shape)
    outputs = ResNet50(weights=None, input_shape=input_shape, classes=classes)(inputs)
    model = keras.Model(inputs, outputs)

    return model
