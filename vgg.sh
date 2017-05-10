#!bin/bash
p=0.5
name=sr_res_vgg_"p=$p"_5_10_16
nohup python -u sr_res_vgg22.py $name $p >> info.$name &
