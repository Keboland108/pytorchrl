#!/bin/bash
DIR=code_examples/train_atari/sac
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
