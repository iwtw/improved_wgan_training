#!/bin/bash
name=srResnet_mse_5_5_21_58
nohup python -u srResnet.py $name >> info.$name &
