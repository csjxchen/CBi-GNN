#!/bin/bash
cd ../excutes
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/cbi_pswarp.py --gpus=1
