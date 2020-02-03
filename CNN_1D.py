from numpy import mean
from numpy import std
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import plot_model
from data_handling import load_dataset


# standardize data
def scale_data(X_train, X_test, X_val):
    # remove overlap: the data is split into windows of 128 time
    # steps, with a 50% overlap. To do it properly we must first
    # remove the duplicated before fitting the StandardScaler()
    cut = int(X_train.shape[1] / 2)
    longX = X_train[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flatTestX = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    flatValidationX = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
    # standardize
    s = StandardScaler()
    # fit on training data
    s.fit(longX)
    # apply to training, test and validation data
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    flatValidationX = s.transform(flatValidationX)
    # reshape
    flatTrainX = flatTrainX.reshape((X_train.shape))
    flatTestX = flatTestX.reshape((X_test.shape))
    flatValidationX = flatValidationX.reshape((X_val.shape))

    return flatTrainX, flatTestX, flatValidationX


# Fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test,X_val, y_val, verbose=1):
    X_train, X_test, X_val = scale_data(X_train, X_test, X_val)

    epochs, batch_size = 15, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    # Multi-headed CNN model

    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # head 3
    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # Concatenate heads
    merged = concatenate([flat1, flat2, flat3])

    dense1 = Dense(85, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # save a plot of the model
    #plot_model(model, show_shapes=True, to_file='data/img_models/multihead_1D_CNN.png')

    # Fit model
    model.fit([X_train, X_train, X_train], y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose,
              validation_data=([X_val, X_val, X_val], y_val), shuffle=True)

    # Evaluate model
    _, accuracy = model.evaluate([X_test, X_test, X_test], y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    # summarize mean and standard deviation
    m, s = mean(scores), std(scores)
    print('Score: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10, verbose=1):
    # load data
    val_split = 0.5
    X_train, X_test, X_val, y_train, y_test, y_val = load_dataset(validation_split=val_split, verbose=0)
    # zero-offset class values
    y_train = y_train - 1
    y_test = y_test - 1
    y_val = y_val - 1
    # one hot encode y
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val, verbose)
        score = score * 100.0
        print('> Rep_%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


if __name__ == '__main__':
    run_experiment(repeats=10)
