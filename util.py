import math
import os
import random
import time

import click
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(spectrogram):
    librosa.display.specshow(np.transpose(spectrogram), cmap="plasma")
    plt.tight_layout()
    plt.savefig("spectrogram.png", bbox_inches=None, pad_inches=0)
    plt.close()


def inference(wav, model, output):
    """
    Converts an input WAV file to an 80-band mel spectrogram, then runs
    inference on the spectrogram using a frozen graph.
    Writes the output to a WAV file.
    """
    data, sr = librosa.core.load(wav, sr=SAMPLE_RATE, mono=True)
    print("Length of audio: {:.2f}s".format(float(len(data))/sr))

    spectrogram = compute_spectrogram(data, sr)
    plot_spectrogram(spectrogram)

    audio = run_wavernn(model, spectrogram, output)
    librosa.output.write_wav(output, audio, sr=SAMPLE_RATE)
    print("Wrote WAV file:", os.path.abspath(output))


def compute_spectrogram(audio, sr):
    """
    Converts audio to an 80-band mel spectrogram.
    Args:
        audio: Raw audio data.
        sr:    Audio sample rate in Hz.
    Returns:
        80-band mel spectrogram, a numpy array of shape [frames, 80].
    """
    spectrogram = librosa.core.stft(audio, n_fft=2048, hop_length=400,
        win_length=1600)
    spectrogram = np.abs(spectrogram)
    spectrogram = np.dot(
        librosa.filters.mel(sr, 2048, n_mels=80, fmin=0, fmax=8000),
        spectrogram)
    spectrogram = np.log(spectrogram*SCALING + 1e-2)
    return np.transpose(spectrogram)