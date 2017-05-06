#!/bin/bash
P=1e-2
name="perceptual_mse_$P""xadv_5_6_15_00"
nohup python -u gan_perceptual.py $name $P >> info.$name &
