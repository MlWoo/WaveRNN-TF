import argparse
import tensorflow as tf
from wavernn.train import wavernn_train
from infolog import log
from hparams import hparams
import os
import infolog
from time import sleep

log = infolog.log


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_output_dir, 'logs_2-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name)
    return log_dir, modified_hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_input_dir', default='')
    parser.add_argument('--base_output_dir', default='/home/lynn/workspace/wumenglin/WaveRNN')
    parser.add_argument('--wavernn_input', default='training_data_dual_channels/train.txt')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--input_dir', default='training_data_dual_channels/',
                        help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--model', default='WaveRNN')
    parser.add_argument('--mode', default='synthesis', help='mode for synthesis')
    parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=3000,
        help='Steps between eval on test data')
    parser.add_argument('--wavernn_train_steps', type=int, default=360000, help='total number of wavenet training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()

    accepted_models = ['WaveRNN', 'Encoder']

    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

    log_dir, hparams = prepare_run(args)

    if args.model == 'WaveRNN':
        wavernn_train(args, log_dir, hparams)
    #elif args.model == 'Encoder':
    #    encoder_train(args, log_dir, hparams)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
    main()

