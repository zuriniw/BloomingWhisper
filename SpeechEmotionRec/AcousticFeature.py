import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

class Spectrogram:
    """Spectrogram (Mel Spectrogram) Features"""
    def __init__(self, input_file, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming", preemph=0.97):
        """
        Initialize
        :param input_file: Input audio file
        :param sr: Sampling rate of the input audio file, default is None
        :param frame_len: Frame length, default is 512 samples (32ms at 16kHz), same as window length
        :param n_fft: Length of the FFT window, default is the same as window length
        :param win_step: Window step, default is moving 2/3 of the frame, 512*2/3=341 samples (21ms at 16kHz)
        :param window: Window type, default is Hamming window
        :param preemph: Pre-emphasis coefficient, default is 0.97
        """
        self.input_file = input_file
        self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
        self.window_len = frame_len
        if n_fft is None:
            self.fft_num = self.window_len
        else:
            self.fft_num = n_fft
        self.hop_length = round(self.window_len * win_step)
        self.window = window

    def get_mel_spectrogram(self, n_mels=128):
        """
        Obtain Mel Spectrogram:
        :param n_mels: Number of Mel filters, default is 128
        :return: Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(self.wave_data, self.sr, n_fft=self.fft_num,
                                                  hop_length=self.hop_length, win_length=self.window_len,
                                                  window=self.window, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec

    def plot(self, show=True):
        """
        Plot Mel Spectrogram
        :param show: Whether to display the image
        :return: None
        """
        mel_spec = self.get_mel_spectrogram()
        librosa.display.specshow(mel_spec, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if show:
            plt.show()

if __name__ == "__main__":
    current_path = os.getcwd()
    wave_file = os.path.join(current_path, "audios/audio_raw.wav")

    # Spectrogram features
    spectrogram_f = Spectrogram(wave_file)
    spectrogram_f.plot()
