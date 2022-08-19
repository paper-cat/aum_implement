import pickle
from abc import ABC

import tensorflow as tf
import numpy as np
import time


# data      : 10000 x 3072 numpy array, 32x32 image, red, green and blue channels
# labels    : 10000 range 0-9, 10 labels

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


class Cifar10Dataset(tf.data.Dataset, ABC):
    def __new__(cls, file_path, shape=(32, 32, 3)):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.int64, tf.dtypes.int64),
            # output_shapes=((shape[0], shape[1] * shape[2] * shape[3]), (shape[0],)),
            output_shapes=((shape[0] * shape[1] * shape[2]), ()),
            args=(file_path,)
        )

    @staticmethod
    def _generator(file_path):
        # 파일 열기
        for i in range(1, 6):
            data_dict = unpickle(file_path.decode('utf-8') + str(i))
            data = data_dict[b'data']
            labels = data_dict[b'labels']

            yield from zip(data, labels)
            # yield data, labels


def map_preproc(data, labels):
    # data = tf.transpose(tf.reshape(tf.convert_to_tensor(data), (10000, 3, 32, 32)), (0, 2, 3, 1))
    data = tf.transpose(tf.reshape(tf.convert_to_tensor(data), (3, 32, 32)), (1, 2, 0))
    data = data / 255
    return data, labels


"""
    fit time 이 1초일때 1 * 6 * 10
    그냥 실행   : 57 초
    AUTOTUNE    : 53 초
    map 사용    : 53 초
    map autotune: 53 초
    cache       : 51 초
    batch 128   : 15 초 
"""


def run_test():
    start_time = time.time()
    dataset = Cifar10Dataset('../data/cifar-10-batches-py/data_batch_').map(map_preproc,
                                                                            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(
        1024)
    # dataset = Cifar10Dataset()

    print('start')
    for epoch_num in range(2):
        print('epoch ', epoch_num)
        for sample in dataset:
            time.sleep(0.0001)

    print('Done With ', time.time() - start_time)


if __name__ == '__main__':
    # test = unpickle('../data/cifar-10-batches-py/data_batch_1')
    run_test()
    pass
