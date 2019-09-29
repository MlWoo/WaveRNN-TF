import numpy as np
import librosa
from datasets import audio
power_2_15 = 32768

def encode_16bits(x):
  return np.clip(x * power_2_15, -power_2_15, power_2_15 - 1).astype(np.int16)


def dual_channels_quantize(x):
  """Mu-Law companding + quantize
  """
  encoded = encode_16bits(x)
  unsigned = encoded + power_2_15
  coarse = unsigned // 256
  fine = unsigned % 256
  print(coarse)
  print(fine)
  return coarse, fine


def combine_signal(coarse, fine):
  signal = coarse * 256 + fine
  #signal -= power_2_15
  signal = signal*2.0/65536 - 1.0
  return signal.astype(np.float32)


def main():
  file_name = "/home/lynn/dataset/LJSpeech-1.1/wavs/LJ001-0001.wav"
  path = "/home/lynn/workspace/wumenglin/WaveRNN/training_data_dual_channels/audio/"
  name = "speech-audio-00001.npy"

  wav = librosa.core.load(file_name, 22050)[0]
  wav = wav / np.abs(wav).max() * 0.999
  print(wav)
  audio_2_channels = dual_channels_quantize(wav)
  out = np.stack(audio_2_channels, axis=1)

  out_dtype = np.int16
  np.save("/home/lynn/workspace/wumenglin/WaveRNN/training_data_dual_channels/speech-audio-00001.npy", out.astype(out_dtype), allow_pickle=False)

  #audio_c = audio_2_channels[0]
  #audio_f = audio_2_channels[1]

  audio_2_channels = np.load("/home/lynn/workspace/wumenglin/WaveRNN/training_data_dual_channels/speech-audio-00001.npy")
  audio_2_channels = audio_2_channels.astype(np.float32)
  audio_c = audio_2_channels[:, 0]
  print(audio_c)
  audio_f = audio_2_channels[:, 1]
  print(audio_f)

  wav_r = combine_signal(audio_c, audio_f)
  librosa.output.write_wav("/home/lynn/workspace/wumenglin/WaveRNN/training_data_dual_channels/audio.wav", wav_r, 22050)


if __name__ == "main":
  main()
