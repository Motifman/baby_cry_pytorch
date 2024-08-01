import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def plot_mels(mel, sr):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


if __name__ == '__main__':
    waveforms = np.load('baby_cry_preprocessed_wave.npy', allow_pickle=True)
    mels = np.load('baby_cry_preprocessed_mel.npy', allow_pickle=True)
    print(waveforms.shape)
    print(mels.shape)

    for i in range(3):
        librosa.display.waveshow(waveforms[i])
        plot_mels(mels[i], sr=16000)
        plt.show()