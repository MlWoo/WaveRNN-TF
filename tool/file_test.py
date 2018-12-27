import os
from wavernn.util import waveplot, inv_mulaw_quantize
from scipy.io import wavfile
import math
import numpy as np

path = "/home/lynn/workspace/wumenglin/WaveRNN/training_data/audio/"
audio_file = "speech-audio-13100.npy"

wav_file = np.load(os.path.join(path, audio_file))
print(wav_file.shape)
sample_output = inv_mulaw_quantize(wav_file)
wav_path = "/home/lynn/workspace/wumenglin/WaveRNN/waveform.wav"
plot_path = "/home/lynn/workspace/wumenglin/WaveRNN/plot.png"
wavfile.write(wav_path, 16000, sample_output)
waveplot(plot_path, sample_output, sample_output, hparams=None, sample_rate=16000)

