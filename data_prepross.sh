#!/bin/bash
python preprocess.py --input_dir=/data/dataset/source_data/mandarin_24k \
                     --output_dir=./ \
                     --dataset=mandarin \
                     --output=mandarin_24k \
                     --n_jobs=40 
