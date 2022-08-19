import tensorflow as tf
import os

from utils.read_cifar10 import Cifar10Dataset, map_preproc
from model.image_model import get_resnet50

base_directory = 'trained_weights'
version = 1

saving_directory = base_directory + '/' + str(version) + '/'
lr = 1e-3
epochs = 10
classes = 10
input_shape = (32, 32, 3)

if os.path.exists(saving_directory) is False:
    os.makedirs(saving_directory)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=saving_directory + 'resnet50_{epoch}',
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=False)

dataset = Cifar10Dataset('data/cifar-10-batches-py/data_batch_').map(map_preproc,
                                                                     num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(
    1024)

model = get_resnet50(input_shape, classes)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss="sparse_categorical_crossentropy")
model.fit(dataset, epochs=10)
