from tensorflow import keras
import librosa
import librosa.display
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

#Loading the trained model
trained_model = keras.models.load_model("neural_net.h5")

mfcc = []
n_mfcc = 13
n_fft=2048
hop_length=512

#Function to change the shape of the mfccs
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

if __name__=="__main__":
# Loading the audio and applying filters

    # path of the audio file
    path = "ex_seven.wav"
    # Loading the signal, default sample rate is 22050(44100/2 -> Nyquist)
    signal, sr = librosa.load(path,sr=22050)

    # Low-High pass filtering of the signal
    nyquist = sr/2.
    b, a = sg.butter(3, [50.0 / nyquist , 8000 / nyquist], 'band')
    signal = sg.filtfilt(b, a, signal)

    #plotting the audio signal
    plt.figure(figsize=(14, 5))
    plt.title('Waveplot')
    librosa.display.waveplot(signal, sr=sr)
    plt.show()

    #Performing FFT
    signal_fft = np.fft.fft(signal) # numpy array with values = number of samples * secs
    #get magnitude, indicates the contribution of each frequency bin to the overall sound
    signal_mag = np.abs(signal_fft)
    freq = np.linspace(0, sr, len(signal_mag))
    # Kratame tis aristera syxnotites gt oi deksia einai idies(Nyquist)
    left_freq = freq[:int(len(freq)/2)]
    left_magn = signal_mag[:int(len(signal_mag)/2)]

    plt.plot(left_freq,left_magn)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    # Epikalyptomena parathyro parathrisis
    signal_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    #from complex -> magnitude
    spectogram = np.abs(signal_stft)
    #The way we precieve loudness is not linear but logarithmic
    log_spec = librosa.amplitude_to_db(spectogram)

    librosa.display.specshow(log_spec, sr=sr, hop_length=hop_length, )
    plt.title("Spectogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()

    #Extract the MFCCs
    mfccs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length)

    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length)
    plt.title("Spectrum")
    plt.xlabel("Time")
    plt.ylabel("MFCCs")
    plt.colorbar()

    plt.show()

#Extracting ZCR and RMS
    #Computing the zero crossing rate of the signal..
    zcr_signal = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)[0]
    rms_signal = librosa.feature.rms(signal, frame_length=2048, hop_length=512)[0]

    #Making the time vector (which will be used as the x-axis) for plotting the zero crossing rate and root mean squared energy
    time = librosa.times_like(zcr_signal, sr=sr)
    plt.figure(figsize=(14, 5))

    plt.plot(time, zcr_signal, color="y")
    plt.title('Zero Crossing Rate')
    plt.xlabel('Time')
    plt.ylabel('ZCR')
    plt.ylim(0, 1)

    plt.show()

    plt.figure(figsize=(14, 10))

    librosa.display.waveplot(signal,sr=sr, alpha=0.5)
    plt.plot(time, rms_signal, color="r")
    plt.ylim((-0.2,0.2))
    plt.title("RMS Energy on top of the speech signal")

#Foreground vs Background
    start = []
    end = []
    flag = False

    for y,z in zip(rms_signal, time):
        if y >= 0.003 and flag==False:
            start.append(z)
            flag = True
        elif y >= 0.003 and flag == True:
            continue
        elif y < 0.003 and flag == True:
            end.append(z)
            flag = False

# Making predictions
    for i in range(len(start)):
        st = int(start[i]*sr)
        en = int(end[i]*sr)
        #sliced signal with voice
        sliced_signal = signal[st:en]
        sliced_signal = librosa.util.normalize(sliced_signal)
        # save mfcc
        s_mfcc = librosa.feature.mfcc(sliced_signal,
                                    sr=sr,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        pad_mfcc = pad2d(s_mfcc,40)
        mfcc.append(pad_mfcc)

    inputs_to_pred = np.array(mfcc)
    inputs_to_pred = np.expand_dims(inputs_to_pred, -1)

    predictions = trained_model.predict(inputs_to_pred, batch_size=32, verbose=0)
    results=np.argmax(predictions,axis=1)

    print(results)