#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python train.py --name=$2 #"disable_swap"
