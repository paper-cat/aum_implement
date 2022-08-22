import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50


def get_resnet50(input_shape, classes):
    inputs = keras.Input(shape=input_shape)
    model = ResNet50(weights=None, input_shape=input_shape, classes=classes)
    outputs = model(inputs)
    model = keras.Model(inputs, outputs)

    return model


def resnet50_logits(input_shape, classes, weight_path):
    inputs = keras.Input(shape=input_shape)
    model = ResNet50(weights=None, input_shape=input_shape, classes=classes)
    outputs = model(inputs)
    model = keras.Model(inputs, outputs)

    model.load_weights(weight_path).expect_partial()
    model.layers[-1].activation = None

    return model
