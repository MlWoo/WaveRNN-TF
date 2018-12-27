from .WaveRNN import WaveRNN
from .modules import Encoder
from warnings import warn

def create_model(name, hparams, teacher_force=False, apples=None, is_development=False):
    if not hparams.cell_type == "GRU_STD":
        if hparams.out_channels != hparams.quantize_channels*2:
            raise RuntimeError(
                "out_channels must equal to 2 times of quantize_chennels if cell_type is 'std_fc'")
    if hparams.encoder_conditional_features and hparams.cin_channels < 0:
        s = "Encoder conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    if name == 'WaveRNN':
        return WaveRNN(hparams, teacher_force, apples, is_development)
    elif name == 'Encoder':
        return Encoder(hparams, apples)
    else:
        raise Exception('Unknow model: {}'.format(name))
