import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import librosa

DATASET_PATH = "recordings"
SAMPLE_RATE = 22050
n_mfcc=13
n_fft=2048
hop_length=512

digit = []
mfcc = []

pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))


def save_mfcc(signal, f, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512):
    s_mfcc = librosa.feature.mfcc(signal,
                                  sr=sr,
                                  n_mfcc=n_mfcc,
                                  n_fft=n_fft,
                                  hop_length=hop_length)

    pad_mfcc = pad2d(s_mfcc, 40)
    mfcc.append(pad_mfcc)
    digit.append(f.split('_')[0])

if __name__=="__main__":

    i = 0
    for f in os.listdir(DATASET_PATH):
        # Load audio file
        file_path = os.path.join(DATASET_PATH, f)
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        signal = librosa.util.normalize(signal)
        # ----Effects for audio data augmentation----
        # Adding white noise to the signal
        if i % 2 == 0 or i % 6 == 0 or i % 4 == 0:
            white_noise = np.random.randn(len(signal))
            wn_signal = signal + 0.005 * white_noise
            save_mfcc(wn_signal, f)
        # Shifting the sound
        if i % 3 == 0 or i % 5 == 0 or i % 7 == 0:
            sh_signal = np.roll(signal, 1600)
            save_mfcc(sh_signal, f)
        # Stretching the sound
        if i % 12 == 0 or i % 13 == 0 or i % 10:
            str_signal = librosa.effects.time_stretch(signal, 1.2)
            if len(signal) > sr:
                str_signal = str_signal[:sr]
                save_mfcc(str_signal, f)
            else:
                str_signal = np.pad(signal, (0, max(0, sr - len(signal))), "constant")
                save_mfcc(str_signal, f)
        if i % 14 == 0 or i % 15 == 0:
            str_signal = librosa.effects.time_stretch(signal, 0.8)
            if len(signal) > sr:
                str_signal = str_signal[:sr]
                save_mfcc(str_signal, f)
            else:
                str_signal = np.pad(signal, (0, max(0, sr - len(signal))), "constant")
                save_mfcc(str_signal, f)

        # save mfcc
        save_mfcc(signal, f)
        i += 1

    inputs = np.array(mfcc)
    targets = keras.utils.to_categorical(np.array(digit))

    inputs, targets = shuffle(inputs, targets, random_state=42)

    inputs = np.expand_dims(inputs, -1)

    # Split dataset to train/test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    print('inputs_train:', inputs_train.shape)
    print('targets_train:', targets_train.shape)
    print('inputs_test:', inputs_test.shape)
    print('targets_test:', targets_test.shape)

    model = keras.Sequential([
        # input layer -> Flatten: Multi-D and flattens it
        keras.layers.Flatten(input_shape=(13, 40, 1)),
        # 1st hidden layer //
        # Relu is very effective for training because it has better convergence and reduced likelihood of vanishing gradient
        keras.layers.Dense(200, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 2nd hidden layer
        keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # 3rd hidden layer
        keras.layers.Dense(50, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        # Softmax is used in the output layer because it distributes the probabillity througout each output node
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile NN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(x=inputs_train, y=targets_train,
                        validation_data=(inputs_test, targets_test),
                        epochs=200,
                        batch_size=32)

    model.summary()

    predictions = model.predict(inputs_test, batch_size=64, verbose=0)
    results = np.argmax(predictions, axis=1)



    print(classification_report(targets_test, keras.utils.to_categorical(results)))

    model.save("neural_net.h5")
    print("Saved the model to disk")

    fig, axs = plt.subplots(ncols=2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()