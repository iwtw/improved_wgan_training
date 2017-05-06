#!/bin/bash
P=1e-3
name="perceptual_mse_$P""xadv_5_6_11_00"
nohup python -u gan_perceptual.py $name $P >> info.$name &
