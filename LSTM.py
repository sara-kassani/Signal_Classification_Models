from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from data_handling import load_dataset
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model


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


# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val, verbose=1):
    X_train, X_test, X_val = scale_data(X_train, X_test, X_val)

    epochs, batch_size = 15, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    # LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units=100))
    model.add(Dense(units=n_outputs, activation="softmax"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='acc',
                                  factor=0.1,
                                  patience=4,
                                  verbose=verbose,
                                  min_delta=0.001,
                                  mode='max')

    # Save an image of the model architecture
    # plot_model(model, show_shapes=True, to_file='data/img_models/LSTM.png')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=verbose, validation_data=(X_val, y_val), callbacks=[reduce_lr])

    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    # summarize mean and standard deviation
    m, s = mean(scores), std(scores)
    print('Score: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10, verbose=1, kernel=((3, 3),)):
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
        score = evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val, verbose=verbose)
        score = score * 100.0
        print('> Rep_%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


if __name__ == '__main__':
    run_experiment(repeats=10)
