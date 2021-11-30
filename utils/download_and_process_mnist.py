import numpy as np
import os
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def resize_images(image_arrays, size=[32, 32]):
    '''将所有图片设置成相同大小，便于统一数据源进行公平地分析比较'''
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i] = np.asarray(resized_image)

    return np.expand_dims(resized_image_arrays, 3)


def download_and_process_mnist():
    if not os.path.exists('./data/mnist'):
        os.makedirs('./data/mnist')

    mnist = input_data.read_data_sets(train_dir='./data/mnist')

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}

    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}

    with open('./data/mnist/train.pkl', 'wb') as f:
        ''' obj——序列化对象，将对象obj保存到文件file中去；
            file——file表示保存到的类文件对象，file必须有write()接口，file可以是一个以’w’打开的文件或者是一个StringIO对象，
                也可以是任何可以实现write()接口的对象；
            protocol——序列化模式，默认是 0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。
                其中，1是老式的二进制协议；2是新二进制协议）。
        '''
        pickle.dump(train, f, protocol=-1)

    with open('./data/mnist/test.pkl', 'wb') as f:
        pickle.dump(test, f, protocol=-1)


if __name__ == "__main__":
    download_and_process_mnist()
