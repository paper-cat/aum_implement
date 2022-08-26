import pickle
from abc import ABC

import tensorflow as tf
import numpy as np
import time
import random

# data      : 10000 x 3072 numpy array, 32x32 image, red, green and blue channels
# labels    : 10000 range 0-9, 10 labels

random_seed = 42
random.seed(random_seed)


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


class Cifar10Dataset(tf.data.Dataset, ABC):
    def __new__(cls, file_path, shape=(32, 32, 3)):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((shape[0] * shape[1] * shape[2]), ()),
            args=(file_path,)
        )

    @staticmethod
    def _generator(file_path):
        # Open Files
        for i in range(1, 6):
            data_dict = unpickle(file_path.decode('utf-8') + str(i))
            data = data_dict[b'data']
            labels = data_dict[b'labels']

            yield from zip(data, labels)


class Cifar10Dataset_noised(tf.data.Dataset, ABC):

    def __new__(cls, file_path, noise_ratio=0.4, shape=(32, 32, 3)):
        cls.noised_idx = [x for x in random.sample([i for i in range(50000)], k=int(50000 * noise_ratio))]
        cls.threshold_idx = [x for x in random.sample([i for i in range(50000)], k=int(50000 / 11))]

        labels = []

        for i in range(1, 6):
            if type(file_path) is not str:
                labels.extend(unpickle(file_path.decode('utf-8') + str(i))[b'labels'])
            else:
                labels.extend(unpickle(file_path + str(i))[b'labels'])

        cls.noised_labels = [labels[x] for x in cls.noised_idx]
        cls.threshold_labels = [labels[x] for x in cls.threshold_idx]

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((shape[0] * shape[1] * shape[2]), ()),
            args=(file_path, cls.noised_idx, cls.threshold_idx)
        ), cls.noised_idx, cls.threshold_idx, cls.noised_labels, cls.threshold_labels

    @staticmethod
    def _generator(file_path, noised_idx, threshold_idx):
        # 파일 열기
        for i in range(1, 6):
            data_dict = unpickle(file_path.decode('utf-8') + str(i))
            data = data_dict[b'data']
            labels = data_dict[b'labels']

            # 원래 정답을 제외한 0~9 중에 랜덤하게 noise 를 줌
            labels = [
                x if 10000 * (i - 1) + x not in noised_idx else random.choice([j for j in range(10) if j != x]) for
                x in labels]

            # threshold idx 적용
            labels = [x if x not in threshold_idx else 10 for x in labels]

            yield from zip(data, labels)

    def get_labels(self):
        return self.noised_idx, self.threshold_idx, self.noised_labels, self.threshold_labels


def map_preproc(data, labels):
    data = tf.transpose(tf.reshape(tf.convert_to_tensor(data), (3, 32, 32)), (1, 2, 0))
    return data, labels


def run_test():
    start_time = time.time()
    dataset = Cifar10Dataset(
        '../data/cifar-10-batches-py/data_batch_'
    ).map(map_preproc, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(1024)

    print('start')
    for epoch_num in range(2):
        print('epoch ', epoch_num)
        for _ in dataset:
            time.sleep(0.0001)

    print('Done With ', time.time() - start_time)


if __name__ == '__main__':
    # test_dataset, noised_idx, threshold_idx = Cifar10Dataset_noised('../data/cifar-10-batches-py/data_batch_')

    print('None')
