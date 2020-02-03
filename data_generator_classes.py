import numpy as np
import keras
import os
from scipy.io import loadmat

"""
#### Keras Data Generators

To feed the scalograms images to the model we also need to create a custom
data_generator. Data_generators are used to load input data from the drive
on small batches as needed when training the model. That way we avoid
running out of RAM memory when working with large data sets. The generators
are defined on the "**data_generator_classes.py**" file. 

More info about keras data generators:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class DataGenerator(keras.utils.Sequence):
    """ Generates input data for the Keras models """

    def __init__(self, list_IDs, labels, batch_size, dim, n_classes, data_dir, n_channels=1, shuffle=True):
        """ Initialization """
        self.dim = dim
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            # To load from .npy files
            X[i,] = np.load(os.path.join(self.data_dir, ID + '.npy'))

            # To load from .mat files
            # X[i,] = loadmat(os.path.join(self.data_dir, ID + '.mat'))['array']

            # Store class
            y[i] = int(self.labels[ID])

        return X, keras.utils.to_categorical(y-1, num_classes=self.n_classes)


#  For the multi-headed 2D_CNN model we need a different data generator.
#  It can be easily done by inheriting everything from the previous data
#  generator class and just modify the __getitem__ function.

# TODO: Refactor the number of heads so the class works with any number of heads and not just 3

class MultiHeadDataGenerator(DataGenerator):
    """ Generates input data for a multi-headed keras model """

    def __init__(self, *args, **kwargs):
        super(MultiHeadDataGenerator, self).__init__(*args, **kwargs)
        # self.n_heads = heads  #TODO

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        X_h1 = X[:, :, :, :3]
        X_h2 = X[:, :, :, 3:6]
        X_h3 = X[:, :, :, 6:]

        return [X_h1, X_h2, X_h3], y

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            # To load from .npy files
            X[i,] = np.load(os.path.join(self.data_dir, ID + '.npy'))

            # To load from .mat files
            # X[i,] = loadmat(os.path.join(self.data_dir, ID + '.mat'))['array']

            # Store class
            y[i] = int(self.labels[ID])

        return X, keras.utils.to_categorical(y-1, num_classes=self.n_classes)
