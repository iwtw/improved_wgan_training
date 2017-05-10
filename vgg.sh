#!bin/bash
p=0.999
name=sr_res_"p=$p"_5_10_15
nohup python -u sr_res_vgg22.py $name $p >> info.$name &
