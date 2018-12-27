from wavernn.models import create_model
from hparams import hparams
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import click
import os
import time

np.set_printoptions(threshold=np.nan)
tf.set_random_seed(1234)

modified_hp = hparams.parse("")

Spec = tf.placeholder(tf.float32, [None, None, modified_hp.num_mels], name='mel')      # [ SEQLEN, CHANNELS ]
Training = tf.placeholder(tf.bool, None, name='is_training')                           # [ BATCHSIZE ]

#tf.enable_eager_execution()

def load_apples(file):
    apples = np.load(file)
    return apples

apples = load_apples('./params/output_parameters.npz')

wavernn_model = create_model('WaveRNN', modified_hp, apples, is_development=True)
#encoder_model = create_model('Encoder', modified_hp, apples)

#seq_length = 100
#spectrogram = np.random.rand(seq_length, hparams.num_mels).astype(dtype=np.float32)


#print(apples['weights_W'])
affine_e = apples['affine_E']
bn_in_e = apples['bn_in_E']
relu1_e = apples['relu1_E']
conv1_e = apples['conv1_E']
spectrogram = apples['input_spec']
residual = apples['residual']
upsampling = apples['upsampling']
moving_var_1 = apples['moving_Var_1']
moving_mean_S = apples['moving_mean_S']

# dump the results of key nodes
Affine_E = "Encoder_1/Affine/Reshape_1:0"
#Affine_E = "Encoder_1/Const:0"
BN_in_E = "Encoder_1/BN_Relu_Conv1D1/BatchNorm/batchnorm/add:0"
Relu_E = "Encoder_1/BN_Relu_Conv1D1/Relu:0"
Conv1_E = "Encoder_1/BN_Relu_Conv1D1/BiasAdd:0"
Residual = "Encoder_1/Residual/add:0"
Upsampleing = "Encoder_1/UpSamplingByRepetition/Reshape:0"
#Moving_mean_S = "Encoder_1/BN_Relu_Conv1D1/BatchNorm/cond/Switch_2:0"
Moving_mean_S = "Inference/Model/TensorArrayStack/TensorArrayGatherV3:0"

OUTPUT_NODE = "Inference/Model/MuLawExpanding/mul_1:0"


# dump the params of the suspected nodes
Moving_Var_1 = "Encoder/BN_Relu_Conv1D1/BatchNorm/moving_variance/read:0"



def run_wavernn_2(model, spectrogram):
    padding = 12
    spectrogram = np.pad(spectrogram, [[padding, padding], [0, 0]], mode='constant')

    is_training = False
    model.initialize(c=Spec, is_training=Training)
    feed_dict = {Spec: [spectrogram], Training: is_training}

    with tf.Session() as session:
        writer = tf.summary.FileWriter("./training_graph", session.graph)
        session.run(tf.global_variables_initializer())
        affine_e_pred, residual_pred, upsampling_pred, bn_in_e_pred, relu1_e_pred, conv1_e_pred, moving_mean_S_pred,\
            = session.run([Affine_E, Residual, Upsampleing, BN_in_E, Relu_E, Conv1_E, Moving_mean_S], feed_dict=feed_dict)

        diff_affine_e = np.abs(affine_e-affine_e_pred)
        diff_residual = np.abs(residual-residual_pred)
        diff_upsampling = np.abs(upsampling-upsampling_pred)

        diff_bn_in_e = np.abs(bn_in_e - bn_in_e_pred)
        diff_relu1_e = np.abs(relu1_e - relu1_e_pred)
        diff_conv1_e = np.abs(conv1_e - conv1_e_pred)
        diff_moving_mean_S = np.abs(moving_mean_S - moving_mean_S_pred)

        return diff_affine_e, diff_residual, diff_upsampling, diff_bn_in_e, diff_relu1_e, diff_conv1_e, diff_moving_mean_S

        #session.run([], feed_dict=feed_dict)

def compare_results(model, spectrogram):

    diff_affine_e, diff_residual, diff_upsampling, diff_bn_in_e, diff_relu1_e, diff_conv1_e, diff_moving_mean_S \
        = run_wavernn_2(wavernn_model, spectrogram)

    max_diff_affine_e = np.max(diff_affine_e)
    max_diff_residual = np.max(diff_residual)
    max_diff_upsampleing = np.max(diff_upsampling)

    max_diff_bn_in_e = np.max(diff_bn_in_e)
    max_diff_relu1_e = np.max(diff_relu1_e)
    max_diff_conv1_e = np.max(diff_conv1_e)
    max_diff_moving_mean_S = np.max(diff_moving_mean_S)

    print('max diff affine_e', max_diff_affine_e)
    print('max diff residual', max_diff_residual)
    print('max diff upsampling', max_diff_upsampleing)

    print('max diff bn_in_e', max_diff_bn_in_e)
    print('max diff relu1_e', max_diff_relu1_e)
    print('max diff conv1_e', max_diff_conv1_e)
    print('max diff moving_mean_S', max_diff_moving_mean_S)


MEL_BANDS = 80
SAMPLE_RATE = 16000
SCALING = 0.185


@click.command()
@click.argument("wav")
@click.option("--output", default="outputs/audio_dev.wav", help="Output WAV audio")
def inference(wav, output):
    """
    Converts an input WAV file to an 80-band mel spectrogram, then runs
    inference on the spectrogram using a frozen graph.
    Writes the output to a WAV file.
    """
    data, sr = librosa.core.load(wav, sr=SAMPLE_RATE, mono=True)
    print("Length of audio: {:.2f}s".format(float(len(data))/sr))

    spectrogram = compute_spectrogram(data, sr)
    print(data.shape, spectrogram.shape)
    plot_spectrogram('spec_raw.png', spectrogram)

    audio = run_wavernn(wavernn_model, spectrogram)
    spectrogram = compute_spectrogram(audio, sr)
    plot_spectrogram('spec_syn.png', spectrogram)

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


def run_wavernn(model, spectrogram):
    padding = 12
    spectrogram = np.pad(spectrogram, [[padding, padding], [0, 0]], mode='constant')

    is_training = False
    model.initialize(c=Spec, is_training=Training)
    feed_dict = {Spec: [spectrogram], Training: is_training}

    with tf.Session() as session:
        writer = tf.summary.FileWriter("./training_graph")
        #session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)
        session.run(tf.global_variables_initializer())
        start_time = time.time()
        audio = session.run(OUTPUT_NODE, feed_dict=feed_dict)
        print(type(audio))
        elapsed = time.time() - start_time
        generated_seconds = audio.size / SAMPLE_RATE
        print("Generated {:.2f}s in {:.2f}s ({:.3f}x realtime) samplepoint {:10d}."
              .format(generated_seconds, elapsed, generated_seconds / elapsed, audio.size))
        return audio

def plot_spectrogram(path, spectrogram):
    librosa.display.specshow(np.transpose(spectrogram), cmap="plasma")
    plt.tight_layout()
    plt.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close()

if __name__ == '__main__':
    inference()
