import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.read_cifar10 import Cifar10Dataset, map_preproc, Cifar10Dataset_noised
from model.image_model import get_resnet50, resnet50_logits

""" Settings """
base_directory = 'trained_weights'
version = 3

saving_directory = base_directory + '/v' + str(version) + '/'
input_shape = (32, 32, 3)
classes = 10

# hyper parameters
val_size = 0.2
lr = 1e-5
epochs = 100
batch_size = 128


def train_model(model_func, train_data, val_data):
    if os.path.exists(saving_directory) is False:
        os.makedirs(saving_directory)

    cp_callback = ModelCheckpoint(
        filepath=saving_directory + 'resnet50_{epoch}',
        save_weights_only=True,
        monitor='loss',
        mode='max',
        save_best_only=False)

    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='auto',
        restore_best_weights='True'
    )

    # model = get_resnet50(input_shape, classes)
    model = model_func(input_shape, classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=['acc'])
    model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[cp_callback, es_callback], shuffle=True)
    model.save_weights(saving_directory + 'resnet50_done')
    print('Finish Training Model!!')


def predict_model(model_func, weight_path, dataset):
    model = model_func(input_shape, classes)
    model.load_weights(weight_path).expect_partial()
    output = model.predict(next(iter(dataset))[0][:1])

    return output


def get_logits(logit_func, weight_path, dataset):
    model = logit_func(input_shape, classes, weight_path)
    output = model.predict(dataset)
    return output


if __name__ == '__main__':
    # Standard Run

    """
    weight_path_ = saving_directory + 'resnet50_done'
    dataset_ = Cifar10Dataset(
        'data/cifar-10-batches-py/data_batch_'
    ).map(map_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()

    dataset_len = len(list(dataset_))

    # split dataset
    dataset_ = dataset_.shuffle(dataset_len, seed=42)
    train_dataset = dataset_.take(int(dataset_len * (1 - val_size))).batch(batch_size)
    val_dataset = dataset_.skip(int(dataset_len * (1 - val_size))).batch(batch_size)

    # train_model(get_resnet50, train_dataset, val_dataset)
    output = get_logits(resnet50_logits, weight_path_, val_dataset)
    print(output)
    """

    # AUM Threshold Run
    classes += 1
    dataset_, noised_idx, threshold_idx, noised_labels, threshold_labels = Cifar10Dataset_noised(
        file_path='data/cifar-10-batches-py/data_batch_')

    dataset_ = dataset_.map(map_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset_len = len(list(dataset_))

    dataset_ = dataset_.shuffle(dataset_len, seed=42)
    train_dataset = dataset_.take(int(dataset_len * (1 - val_size))).batch(batch_size)
    val_dataset = dataset_.skip(int(dataset_len * (1 - val_size))).batch(batch_size)

    # train_model(get_resnet50, train_dataset, val_dataset)

    # get aums of threshold_idx
    dataset_ = dataset_.batch(1)
    epochs = max([int(x.split('_')[1].split('.')[0]) for x in os.listdir(saving_directory) if
                  ('resnet50' in x) and ('of' not in x) and ('done' not in x)])

    print('calc aum values,,,')
    aums = []
    for epoch in tqdm(range(1, epochs + 1)):
        aum_ep = []
        weight_path_ = saving_directory + 'resnet50_' + str(epoch)
        logits = get_logits(resnet50_logits, weight_path_, dataset_)

        for i, (_, label) in enumerate(iter(dataset_)):
            threshold_item = False
            if i in threshold_idx:
                threshold_item = True

            logit = logits[i]
            if threshold_item:
                label = threshold_labels[i]
            max_val = np.argmax(logit)

            # max = threshold label
            if threshold_item:
                if max_val == 10:
                    aum_ep.append(logit[10] - logit[int(np.argsort(logit, axis=0)[-2])])
                else:
                    aum_ep.append(logit[max_val] - logit[10])
            else:
                if max_val == label:
                    aum_ep.append(logit[label] - logit[int(np.argsort(logit, axis=0)[-2])])
                else:
                    aum_ep.append(logit[max_val] - logit[label])

        aums.append(aum_ep)

    print('calc threshold value...')
    aum_result = []
    for i, idx in enumerate(threshold_idx):
        item_aum = []
        for epoch in aums:
            item_aum.append(epoch[idx])

        aum_result.append(np.average(item_aum))

    print('Threshold Aum Value is ', np.percentile(aum_result, 99))

    # value : 0.53
