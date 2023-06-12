#!/bin/bash
source /home/xavier/development/env/bin/activate

DIR=code_examples/train_pybullet/sac
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml

deactivate