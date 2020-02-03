from numpy import mean
from numpy import std
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import multiprocessing
from keras.callbacks import ReduceLROnPlateau, History
from keras.optimizers import Adam
from keras.models import Sequential
from data_handling import create_ids_labels
from data_generator_classes import DataGenerator
from keras.utils import plot_model
history = History()

data_dir = 'data/data_scalograms/'


def build_model(dim_img=(224, 224), n_channels=3):
    input_shape = dim_img + (n_channels,)

    model = Sequential()
    model.add(Conv2D(filters=102, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=102, kernel_size=5, activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())

    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=6, activation="softmax"))

    return model


def evaluate_model(verbose=1):
    # Model parameters
    epochs = 15
    batch_size = 32
    optimizer = Adam(lr=0.01)
    # Reduce the learning rate once the learning stagnates, it is good in order
    # try to scratch those last decimals of accuracy.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  verbose=1,
                                  min_delta=1e-4,
                                  mode='max')
    # Data generator parameters
    data_dir_train = data_dir + 'train'
    data_dir_validation = data_dir + 'test'

    dim = (224, 224)
    n_channels = 9
    params = {'dim': dim,
              'batch_size': batch_size,
              'n_classes': 6,
              'n_channels': n_channels,
              'shuffle': True}

    ids_train, labels_train = create_ids_labels(data_dir_train)
    ids_test, labels_test = create_ids_labels(data_dir_validation)

    # Generators
    training_generator = DataGenerator(ids_train, labels_train, data_dir=data_dir_train, **params)
    validation_generator = DataGenerator(ids_test, labels_test, data_dir=data_dir_validation, **params)

    # Model
    model = build_model(dim_img=dim, n_channels=n_channels)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        verbose=verbose,
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=multiprocessing.cpu_count(),  # max number of workers
                        callbacks=[reduce_lr])

    # Evaluate model
    _, accuracy = model.evaluate_generator(generator=validation_generator,
                                           workers=multiprocessing.cpu_count(),
                                           use_multiprocessing=True,
                                           verbose=verbose)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    # summarize mean and standard deviation
    m, s = mean(scores), std(scores)
    print('Score: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10, verbose=1):
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(verbose)
        score = score * 100.0
        print('> Rep_%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


if __name__ == '__main__':
    run_experiment(repeats=10)
