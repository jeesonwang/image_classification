#!/bin/bash
export CUDA_VISIBLE_DEVICES='3'

python3 main.py --arch resnet --depth 56 --save save/cifar100-resnet-56 --data cifar100 --data_aug