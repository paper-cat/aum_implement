import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import time
import pickle
import random

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.read_cifar10 import Cifar10Dataset, map_preproc, Cifar10Dataset_noised
from model.image_model import get_resnet50, resnet50_logits

""" Settings """

"""
v1~5 : adam
v6 : sgd, lr 0.1 val 0.1
"""
base_directory = 'trained_weights'
version = 6

saving_directory = base_directory + '/v' + str(version) + '/'
input_shape = (32, 32, 3)
classes = 10

# hyper parameters
val_size = 0.1
lr = 1e-1
epochs = 100
batch_size = 64


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
        min_delta=0.001,
        restore_best_weights='True'
    )

    model = model_func(input_shape, classes)
    model.compile(  # optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=['acc']
    )
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
    # # # # # Standard Train # # # # #
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

    # Train model
    train_model(get_resnet50, train_dataset, val_dataset)
    """
    # #### AUM Threshold Run #### #

    # ## Get Noised Data

    classes += 1
    dataset_, noised_idx, threshold_idx, noised_labels, threshold_labels = Cifar10Dataset_noised(
        file_path='data/cifar-10-batches-py/data_batch_', noise_ratio=0.2)

    dataset_ = dataset_.map(map_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset_len = len(list(dataset_))

    dataset_ = dataset_.shuffle(dataset_len, seed=42)
    train_dataset = dataset_.take(int(dataset_len * (1 - val_size))).batch(batch_size)
    val_dataset = dataset_.skip(int(dataset_len * (1 - val_size))).batch(batch_size)

    # ## Train
    train_model(get_resnet50, train_dataset, val_dataset)

    # ## get aums of threshold_idx
    dataset_ = dataset_.batch(batch_size)
    epochs = max([int(x.split('_')[1].split('.')[0]) for x in os.listdir(saving_directory) if
                  ('resnet50' in x) and ('of' not in x) and ('done' not in x)])

    print('calc aum values,,,')
    aums = []
    for epoch in tqdm(range(1, epochs + 1)):
        aum_ep = []
        weight_path_ = saving_directory + 'resnet50_' + str(epoch)
        logits = get_logits(resnet50_logits, weight_path_, dataset_)
        for i, (_, batch_label) in enumerate(iter(dataset_)):

            for j, label in enumerate(batch_label):
                label = int(label)
                threshold_item = False
                idx = j + i * batch_size
                if idx in threshold_idx:
                    threshold_item = True

                logit = logits[idx]
                if threshold_item:
                    label = threshold_labels[threshold_idx.index(idx)]
                max_val = np.argmax(logit)

                if max_val == label:
                    aum_ep.append(logit[label] - logit[int(np.argsort(logit, axis=0)[-2])])
                else:
                    aum_ep.append(logit[label] - logit[max_val])

        aums.append(aum_ep)

    # # # Saving aum values for later
    with open('aum_files/v' + str(version) + '_aums.pkl', 'wb') as f:
        pickle.dump(aums, f)

    with open('aum_files/v' + str(version) + '_aums.pkl', 'rb') as f:
        aums = pickle.load(f)

    print('calc threshold value...')
    aum_result = []
    for i, idx in enumerate(threshold_idx):
        item_aum = []
        for epoch in aums:
            item_aum.append(epoch[idx])

        aum_result.append(np.average(item_aum))

    th_val = np.percentile(aum_result, 99)
    print('Threshold Aum Value is ', th_val)
    # value : -2.34

    aum_result = []
    for item in tqdm(range(len(aums[0]))):
        item_aum = []
        for epoch, aum in enumerate(aums):
            item_aum.append(aum[item])
        aum_result.append(np.average(item_aum))

    print('Calc aum values Done')

    mislabeled = []
    for i, aum in tqdm(enumerate(aum_result)):
        if i in threshold_idx:
            pass
        elif aum < th_val:
            mislabeled.append(i)

    noised_idx.extend(threshold_idx)
    noised_w_th = list(sorted(set(noised_idx)))

    print('detected mislabeled length', len(mislabeled))
    print("Finding Mislabeled Done")
    right = 0
    wrong = 0
    for item in tqdm(mislabeled):
        if item in noised_w_th:
            right += 1
        else:
            wrong += 1

    print('Find Mislabeled well, ', right)
    print('Find Mislabeled wrong, ', wrong)
    print('Missing noise', len(noised_w_th) - right)
    print('Detecting Ratio,', right / len(noised_idx))
