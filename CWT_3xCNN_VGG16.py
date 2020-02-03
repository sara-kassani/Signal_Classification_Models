from numpy import mean
from numpy import std
from keras.layers import Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import VGG16
import multiprocessing
from keras.callbacks import ReduceLROnPlateau, History
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.models import Model
from data_handling import create_ids_labels
from data_generator_classes import MultiHeadDataGenerator
from keras.utils import plot_model
history = History()

data_dir = 'data/data_scalograms/'


def build_model(dim_img=(224, 224), n_channels=3):
    n_heads = 3
    input_shape = dim_img + (int(n_channels / n_heads), )

    # Import model discarding the last layers (output layers).
    head1 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    head2 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    head3 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # There can not be layers with the same name in the model,
    # so we need to rename the layers with unique names.
    for Id, head in enumerate([head1, head2, head3], start=1):
        for layer in head.layers:
            new_name = f"head{Id}_" + layer.name
            layer.name = new_name

    # Concatenate the 3 VGG16 convolutional heads
    x = concatenate([head1.output, head2.output, head3.output])

    x = GlobalAveragePooling2D()(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)

    # Create model based on our architecture
    model = Model(inputs=[head1.input, head2.input, head3.input], outputs=outputs)
    # plot_model(model, show_shapes=False, to_file='data/img_models/x3_CWT_CNN.png')

    # Set which layer from the base_model we want to retrain.
    static_layers = len(head1.layers)  # This sets all layers to not trainable
    for head in [head1, head2, head3]:
        for layer in head.layers[:static_layers]:
            layer.trainable = False

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
                                  patience=7,
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
    training_generator = MultiHeadDataGenerator(ids_train, labels_train, data_dir=data_dir_train, **params)
    validation_generator = MultiHeadDataGenerator(ids_test, labels_test, data_dir=data_dir_validation, **params)

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







