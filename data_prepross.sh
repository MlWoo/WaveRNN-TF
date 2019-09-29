#!/bin/bash
python preprocess.py --input_dir=/home/lynn/dataset/biaobei_24k/ \
	                   --output_dir=/home/lynn/workspace/wumenglin/WaveRNN/ \
										 --dataset=mandarin \
										 --output=mandarin_24k \
										 --n_jobs=12 
