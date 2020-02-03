import numpy as np
from scipy.io import savemat, loadmat
from collections import Counter
import os
import re


def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data


def read_labels(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def load_dataset(verbose=1, validation_split=None):
    INPUT_FOLDER_TRAIN = 'data/UCI HAR Dataset/train/Inertial Signals/'
    INPUT_FOLDER_TEST = 'data/UCI HAR Dataset/test/Inertial Signals/'
    LABELFILE_TRAIN = 'data/UCI HAR Dataset/train/y_train.txt'
    LABELFILE_TEST = 'data/UCI HAR Dataset/test/y_test.txt'

    INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                         'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                         'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

    INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                        'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                        'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

    train_signals, test_signals = [], []

    for input_file in INPUT_FILES_TRAIN:
        signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    for input_file in INPUT_FILES_TEST:
        signal = read_signals(INPUT_FOLDER_TEST + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = read_labels(LABELFILE_TRAIN)
    test_labels = read_labels(LABELFILE_TEST)

    [no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
    [no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)

    if verbose > 0:
        print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train,
                                                                                                       no_steps_train,
                                                                                                       no_components_train))
        print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test,
                                                                                                      no_steps_test,
                                                                                                      no_components_test))
        print("The train dataset contains {} labels, with the following distribution:\n {}".format(
            np.shape(train_labels)[0], Counter(train_labels[:])))
        print("The test dataset contains {} labels, with the following distribution:\n {}".format(
            np.shape(test_labels)[0], Counter(test_labels[:])))

    train_signals, train_labels = randomize(train_signals, np.array(train_labels))
    test_signals, test_labels = randomize(test_signals, np.array(test_labels))

    # To save memory
    train_signals = train_signals.astype('float32')
    test_signals = test_signals.astype('float32')

    # TODO
    # Validation
    if validation_split:
        index = int(test_signals.shape[0] * validation_split)
        validation_signals = test_signals[: index, :, :]
        validation_labels = test_labels[: index]
        evaluation_signals = test_signals[index:, :, :]
        evaluation_labels = test_labels[index:]
        if verbose > 0:
            print(
                "The test data set has been splitted in a validation set ({} samples) and a evaluation set ({} samples)".format(
                    np.shape(validation_labels)[0], np.shape(evaluation_labels)[0]))

        return train_signals, validation_signals, evaluation_signals, train_labels, validation_labels, evaluation_labels

    return train_signals, test_signals, train_labels, test_labels


# Save the array with the right format and shape as a .mat file to
# read it easily on MATLAB when needed.
def save_mat(data_dir='data/UCI HAR Dataset/'):
    X_train, X_test, y_train, y_test = load_dataset()
    X_train = X_train.astype('float64')  # float64 == MATLAB double
    X_test = X_test.astype('float64')
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    savemat(data_dir + 'HAR_signals_labels.mat', mdict={
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })
    print("Saved .mat")


def create_ids_labels(data_dir):
    """ Create the dictionary necessary for the keras data generator. """
    labels_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npy"):
                key = re.sub(r"\.npy", "", file)  # <-- The key acts as the ID of the sample
                value = int(key[0])  # <-- The value is the actual label of the sample
                labels_dict[key] = value
    ids = list(labels_dict.keys())
    return ids, labels_dict


def mat2npy(data_dir):
    """ Converts array .mat files to .npy """
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                try:
                    arr = loadmat(os.path.join(root, file))['array']
                    filename = re.sub(".mat", ".npy", file)
                    np.save(arr=arr, file=os.path.join(root, filename))
                except Exception as e:
                    print(file)
                    print(e)
