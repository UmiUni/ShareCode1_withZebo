import pickle
import random
import numpy as np

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NB_CLASSES = 10

class cifar10(object):
    def __init__(self, data_dir='cifar-10-batches-py/', batch_size=100, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.data = []
        self.labels = []
        self._read_data()
        self._shuffle()
        self.i = 0
        self.batch_size = batch_size
        self.length = len(self.labels)

    def _read_data(self):
        if self.subset == 'train':
            for i in range(1, 6):
                filename = 'data_batch_%d' % i
                with open(self.data_dir + filename, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                self.data.extend(list(data_dict[b'data']))
                self.labels.extend(data_dict[b'labels'])
        elif self.subset == 'test':
            filename = 'test_batch'
            with open(self.data_dir + filename, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            self.data.extend(list(data_dict[b'data']))
            self.labels.extend(data_dict[b'labels'])
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def _shuffle(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)

    def next_batch(self):
        if self.i + self.batch_size > self.length:
            self.i = 0
        images = np.array([self.preprocess(image) for image in self.data[self.i: self.i + self.batch_size]])
        labels = np.array(self.labels[self.i: self.i + self.batch_size])
        labels = np.eye(NB_CLASSES)[labels]

        self.i += self.batch_size

        return images, labels

    def preprocess(self, image):
        image = np.transpose(np.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1,2,0]).astype(np.float32)
        image /= 255.
        image -= image.mean()
        image /= image.std()
        if self.subset == 'train' and self.use_distortion:
            #TODO distortion
            pass
        return image

    @property    
    def num_iter_per_epoch(self):
        return self.length // self.batch_size
